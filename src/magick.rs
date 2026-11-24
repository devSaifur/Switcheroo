use crate::temp::get_temporary_file_path;
use crate::{color::Color, filetypes::FileType, window::ResizeFilter, window::StripType};
use gettextrs::gettext;
use itertools::Itertools;
use shared_child::SharedChild;
use std::ffi::OsStr;
use std::io::Read;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use tempfile::TempDir;

/// Conservative per-job RAM budget (in MiB) for heavy encoders such as AVIF,
/// HEIC, and JXL, which can easily allocate 500 MiB–1.2 GiB per process.
const CONVERSION_MEM_BUDGET_PER_JOB_MIB: u64 = 1536;

/// Lighter per-job RAM budget (in MiB) for `magick identify` metadata probes.
const IDENTIFY_MEM_BUDGET_PER_JOB_MIB: u64 = 512;

/// Minimum per-process `ImageMagick` pixel-cache memory limit (in MiB).
const MIN_MEMORY_LIMIT_MIB: u64 = 256;

/// Maximum per-process disk cache size (in GiB) so runaway operations cannot fill storage.
const MAX_DISK_LIMIT_GIB: u64 = 4;

/// Maximum allowed resize percentage in `%` mode to prevent accidental pixel-dimension entry.
pub const MAX_RESIZE_PERCENTAGE: usize = 500;

/// Maximum allowed width or height in `px` mode.
pub const MAX_RESIZE_PIXELS: usize = 16_384;

/// Maximum resulting output pixels per image (250 megapixels) to avoid multi-gigabyte caches.
pub const MAX_OUTPUT_PIXELS: u64 = 250_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeValidationError {
    InvalidPercentage,
    PercentageTooLarge,
    InvalidPixels,
    OutputTooLarge,
}

pub fn validate_resize_inputs(
    is_percentage: bool,
    width_text: &str,
    height_text: &str,
    image_dimensions: &[(usize, usize)],
) -> Result<ResizeArgument, ResizeValidationError> {
    let parse_positive = |text: &str| text.trim().parse::<usize>().ok().filter(|&v| v > 0);

    if is_percentage {
        let (Some(width), Some(height)) =
            (parse_positive(width_text), parse_positive(height_text))
        else {
            return Err(ResizeValidationError::InvalidPercentage);
        };

        if width > MAX_RESIZE_PERCENTAGE || height > MAX_RESIZE_PERCENTAGE {
            return Err(ResizeValidationError::PercentageTooLarge);
        }

        for &(img_w, img_h) in image_dimensions {
            let out_w = ((img_w as u64).saturating_mul(width as u64) / 100).max(1);
            let out_h = ((img_h as u64).saturating_mul(height as u64) / 100).max(1);
            if out_w.saturating_mul(out_h) > MAX_OUTPUT_PIXELS {
                return Err(ResizeValidationError::OutputTooLarge);
            }
        }

        Ok(ResizeArgument::Percentage { width, height })
    } else {
        let (Some(width), Some(height)) =
            (parse_positive(width_text), parse_positive(height_text))
        else {
            return Err(ResizeValidationError::InvalidPixels);
        };

        if width > MAX_RESIZE_PIXELS || height > MAX_RESIZE_PIXELS {
            return Err(ResizeValidationError::InvalidPixels);
        }

        if (width as u64).saturating_mul(height as u64) > MAX_OUTPUT_PIXELS {
            return Err(ResizeValidationError::OutputTooLarge);
        }

        Ok(ResizeArgument::ExactPixels { width, height })
    }
}

fn magick_temporary_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    let dir = base.join("switcheroo").join("magick-tmp");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Concurrency and per-process resource caps applied to `magick` invocations so
/// batch conversions do not saturate every CPU core or trigger the OOM killer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceLimits {
    /// Maximum number of `magick` processes allowed to run concurrently.
    pub max_concurrent_jobs: usize,
    /// Maximum number of threads each `magick` process may use.
    pub threads_per_job: usize,
    /// Optional per-process memory/map limit (in MiB) for `ImageMagick`'s pixel cache.
    pub memory_limit_mib: Option<u64>,
}

impl ResourceLimits {
    pub fn detect(job_count: usize) -> Self {
        Self::from_system(job_count, available_cpus(), available_memory_mib())
    }

    pub fn detect_for_identify(job_count: usize) -> Self {
        Self::for_identify(job_count, available_cpus(), available_memory_mib())
    }

    pub fn from_system(
        job_count: usize,
        cpu_count: usize,
        available_mem_mib: Option<u64>,
    ) -> Self {
        let cpu_budget = cpu_budget_for_cores(cpu_count);
        let mem_jobs = available_mem_mib.map_or_else(
            || cpu_budget.min(4),
            |mib| {
                usize::try_from(mib / CONVERSION_MEM_BUDGET_PER_JOB_MIB)
                    .unwrap_or(usize::MAX)
                    .max(1)
            },
        );

        let max_concurrent_jobs = cpu_budget.min(mem_jobs).max(1);
        let active_jobs = job_count.max(1).min(max_concurrent_jobs);
        let threads_per_job = (cpu_budget / active_jobs).max(1);
        let memory_limit_mib = available_mem_mib.map(|mib| {
            let usable_mib = (mib * 3) / 5;
            (usable_mib / active_jobs as u64).max(MIN_MEMORY_LIMIT_MIB)
        });

        Self {
            max_concurrent_jobs,
            threads_per_job,
            memory_limit_mib,
        }
    }

    pub fn for_identify(
        job_count: usize,
        cpu_count: usize,
        available_mem_mib: Option<u64>,
    ) -> Self {
        let cpu_budget = cpu_budget_for_cores(cpu_count);
        let mem_jobs = available_mem_mib.map_or_else(
            || cpu_budget.min(4),
            |mib| {
                usize::try_from(mib / IDENTIFY_MEM_BUDGET_PER_JOB_MIB)
                    .unwrap_or(usize::MAX)
                    .max(1)
            },
        );

        let max_concurrent_jobs = cpu_budget.min(mem_jobs).clamp(1, 8);
        let active_jobs = job_count.max(1).min(max_concurrent_jobs);
        let memory_limit_mib = available_mem_mib.map(|mib| {
            let usable_mib = (mib * 3) / 5;
            (usable_mib / active_jobs as u64).max(MIN_MEMORY_LIMIT_MIB)
        });

        Self {
            max_concurrent_jobs,
            threads_per_job: 1,
            memory_limit_mib,
        }
    }

    pub fn apply_to_command(self, command: &mut Command) {
        let threads = self.threads_per_job.to_string();
        let disk_limit = format!("{MAX_DISK_LIMIT_GIB}GiB");
        command
            .env("MAGICK_THREAD_LIMIT", &threads)
            .env("OMP_NUM_THREADS", &threads)
            .env("MAGICK_DISK_LIMIT", &disk_limit);

        if let Some(memory_mib) = self.memory_limit_mib {
            let limit = format!("{memory_mib}MiB");
            command
                .env("MAGICK_MEMORY_LIMIT", &limit)
                .env("MAGICK_MAP_LIMIT", &limit);
        }

        if let Some(tmp_dir) = magick_temporary_dir() {
            command.env("MAGICK_TEMPORARY_PATH", tmp_dir);
        }
    }

    pub fn apply_to_tokio_command(self, command: &mut tokio::process::Command) {
        let threads = self.threads_per_job.to_string();
        let disk_limit = format!("{MAX_DISK_LIMIT_GIB}GiB");
        command
            .env("MAGICK_THREAD_LIMIT", &threads)
            .env("OMP_NUM_THREADS", &threads)
            .env("MAGICK_DISK_LIMIT", &disk_limit);

        if let Some(memory_mib) = self.memory_limit_mib {
            let limit = format!("{memory_mib}MiB");
            command
                .env("MAGICK_MEMORY_LIMIT", &limit)
                .env("MAGICK_MAP_LIMIT", &limit);
        }

        if let Some(tmp_dir) = magick_temporary_dir() {
            command.env("MAGICK_TEMPORARY_PATH", tmp_dir);
        }
    }
}

const fn cpu_budget_for_cores(cpu_count: usize) -> usize {
    if cpu_count <= 2 {
        1
    } else {
        (cpu_count * 3) / 4
    }
}

fn available_cpus() -> usize {
    std::thread::available_parallelism().map_or(1, NonZeroUsize::get)
}

fn available_memory_mib() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    parse_meminfo_available_mib(&meminfo)
}

fn parse_meminfo_available_mib(meminfo: &str) -> Option<u64> {
    let mut mem_total_kib = None;

    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:")
            && let Some(kib) = rest
                .split_whitespace()
                .next()
                .and_then(|v| v.parse::<u64>().ok())
        {
            return Some(kib / 1024);
        } else if let Some(rest) = line.strip_prefix("MemTotal:")
            && let Some(kib) = rest
                .split_whitespace()
                .next()
                .and_then(|v| v.parse::<u64>().ok())
        {
            mem_total_kib = Some(kib);
        }
    }

    mem_total_kib.map(|kib| kib / 2 / 1024)
}

pub async fn count_frames(
    path: &Path,
    limits: ResourceLimits,
) -> Result<(NonZeroUsize, Option<(usize, usize)>), ()> {
    let mut command = tokio::process::Command::new("magick");
    limits.apply_to_tokio_command(&mut command);
    let output = command
        .stdout(std::process::Stdio::piped())
        .arg("identify")
        .arg(path)
        .output()
        .await
        .map_err(|_| ())?;

    let output_string = std::str::from_utf8(&output.stdout).map_err(|_| ())?;

    let mut lines = output_string.lines();
    // No output at all means ImageMagick could not identify the file: it is
    // corrupted or relies on a delegate library that is missing (e.g. libjxl
    // for JXL, libwebp for WebP). Report it as an error rather than a valid
    // image with zero frames.
    let first_line = lines.next().ok_or(())?;
    // One line per frame; the first line also carries the dimensions.
    let count = NonZeroUsize::MIN.saturating_add(lines.count());
    let dimensions = regex::Regex::new(r" \d+x\d+ ")
        .unwrap()
        .find(first_line)
        .and_then(|regex_match| {
            let dimensions = regex_match
                .as_str()
                .trim()
                .split('x')
                .map(str::parse::<usize>)
                .collect_vec();
            match dimensions[..] {
                [Ok(width), Ok(height)] => Some((width, height)),
                _ => None,
            }
        });
    Ok((count, dimensions))
}

pub trait MagickArgument {
    fn get_argument(&self) -> Vec<String>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResizeArgument {
    Percentage { width: usize, height: usize },
    ExactPixels { width: usize, height: usize },
}

impl Default for ResizeArgument {
    fn default() -> Self {
        Self::Percentage {
            width: 100,
            height: 100,
        }
    }
}

impl MagickArgument for ResizeFilter {
    fn get_argument(&self) -> Vec<String> {
        self.as_display_string()
            .map_or_else(std::vec::Vec::new, |f| {
                vec!["-filter".to_string(), f.to_owned()]
            })
    }
}

impl MagickArgument for StripType {
    fn get_argument(&self) -> Vec<String> {
        match self {
            Self::StripAll => vec!["-auto-orient".to_string(), "-strip".to_string()],
            Self::None => vec![],
        }
    }
}

impl MagickArgument for ResizeArgument {
    fn get_argument(&self) -> Vec<String> {
        match self {
            Self::Percentage {
                width: 100,
                height: 100,
            } => vec![],
            Self::Percentage { width, height } => {
                vec!["-resize".to_owned(), format!("{width}%x{height}%")]
            }
            Self::ExactPixels { width, height } => {
                vec!["-resize".to_owned(), format!("{width}x{height}!")]
            }
        }
    }
}

impl<T> MagickArgument for Option<T>
where
    T: MagickArgument,
{
    fn get_argument(&self) -> Vec<String> {
        self.as_ref()
            .map_or_else(std::vec::Vec::new, MagickArgument::get_argument)
    }
}

#[derive(Debug)]
pub struct IndividualConvertJob {
    pub input_file: PathBuf,
    pub input_file_type: FileType,
    pub output_stem: String,
}

impl IndividualConvertJob {
    pub fn get_magick_job(
        self,
        temporary_directory: &TempDir,
        output_file_type: FileType,
        pdf_dpi: usize,
        magick_arguments: &DefaultMagickArguments,
    ) -> MagickConvertJob {
        generate_job(
            self.input_file,
            self.input_file_type,
            get_temporary_file_path(
                temporary_directory,
                &JobFile::new(output_file_type, Some(self.output_stem)),
            ),
            output_file_type,
            pdf_dpi,
            magick_arguments,
        )
    }
}

pub struct DefaultMagickArguments {
    pub background: Color,
    pub quality: usize,
    pub filter: Option<ResizeFilter>,
    pub resize_arg: ResizeArgument,
    pub strip: StripType,
}

#[derive(Debug, Clone)]
pub struct MagickConvertJob {
    pub input_file: PathBuf,
    pub output_file: PathBuf,
    pub background: Color,
    pub quality: usize,
    pub first_frame: bool,
    pub filter: Option<ResizeFilter>,
    pub strip: StripType,
    pub resize_arg: ResizeArgument,
    pub density: Option<usize>,
    pub remove_alpha: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct JobFile {
    pub id: usize,
    pub desired_name: Option<String>,
    pub file_extension: FileType,
}

static FILE_COUNT: AtomicUsize = AtomicUsize::new(0);

impl JobFile {
    pub fn new(file_extension: FileType, desired_name: Option<String>) -> Self {
        let id = FILE_COUNT.fetch_add(1, Ordering::SeqCst) + 1;
        Self {
            id,
            desired_name,
            file_extension,
        }
    }

    pub fn from_clipboard() -> Self {
        let id = FILE_COUNT.fetch_add(1, Ordering::SeqCst) + 1;
        Self {
            id,
            desired_name: Some(format!("{}.png", gettext("Pasted Image"))),
            file_extension: FileType::Png,
        }
    }

    pub fn as_filename(&self) -> String {
        self.desired_name.as_ref().map_or_else(
            || {
                format!(
                    "TEMPORARY_SWITCHEROO_{}.{}",
                    self.id,
                    self.file_extension.as_extension()
                )
            },
            std::borrow::ToOwned::to_owned,
        )
    }
}

impl MagickConvertJob {
    pub const fn from_default_arguments(
        input_file: PathBuf,
        output_file: PathBuf,
        density: Option<usize>,
        first_frame: bool,
        remove_alpha: bool,
        default_arguments: &DefaultMagickArguments,
    ) -> Self {
        Self {
            input_file,
            output_file,
            background: default_arguments.background,
            quality: default_arguments.quality,
            first_frame,
            filter: default_arguments.filter,
            strip: default_arguments.strip,
            resize_arg: default_arguments.resize_arg,
            density,
            remove_alpha,
        }
    }

    pub fn get_command(&self) -> Command {
        let mut command = Command::new("magick");

        dbg!(self);

        let input_file_ext = self
            .input_file
            .extension()
            .and_then(OsStr::to_str)
            .unwrap_or("")
            .split('[')
            .next()
            .unwrap_or("")
            .to_lowercase();

        let (resize_arg, size_arg) = match input_file_ext.as_str() {
            "svg" => (
                vec![],
                match self.resize_arg {
                    ResizeArgument::ExactPixels { width, height } => {
                        vec!["-size".to_owned(), format!("{width}x{height}")]
                    }
                    ResizeArgument::Percentage { width, height: _ } => {
                        let all_pixels = width as f64 / 100.0;
                        vec![
                            "-density".to_owned(),
                            ((all_pixels * 96.0) as usize).to_string(),
                        ]
                    }
                },
            ),
            "pdf" => (
                self.resize_arg.get_argument(),
                self.density.map_or_else(std::vec::Vec::new, |density| {
                    vec!["-density".to_owned(), density.to_string()]
                }),
            ),
            _ => (self.resize_arg.get_argument(), vec![]),
        };

        dbg!(&resize_arg);
        dbg!(&size_arg);

        if self.first_frame {
            command
                .args(size_arg)
                .args(["-background", &self.background.as_hex_string()])
                .arg(self.input_file.clone())
                .arg("-flatten");

            if self.remove_alpha {
                command.arg("-alpha").arg("off");
            }

            command
                .args(["-quality".to_string(), self.quality.to_string()])
                .args(self.filter.get_argument())
                .args(resize_arg)
                .args(self.strip.get_argument())
                .arg(self.output_file.clone());
        } else {
            command
                .arg(self.input_file.clone())
                .arg("-coalesce")
                .args(vec![
                    "-fill",
                    &self.background.as_hex_string(),
                    "-opaque",
                    "none",
                ])
                .args(vec!["-quality".to_string(), self.quality.to_string()])
                .args(self.filter.get_argument())
                .args(resize_arg)
                .arg(self.output_file.clone());
        }

        command.stdout(Stdio::piped()).stderr(Stdio::piped());

        command
    }
}

pub const fn generate_job(
    input_path: PathBuf,
    input_type: FileType,
    output_path: PathBuf,
    output_type: FileType,
    pdf_dpi: usize,
    default_arguments: &DefaultMagickArguments,
) -> MagickConvertJob {
    match (input_type, output_type) {
        (FileType::Pdf, _) => MagickConvertJob::from_default_arguments(
            input_path,
            output_path,
            Some(pdf_dpi),
            false,
            false,
            default_arguments,
        ),
        (input, output) if input.supports_animation() && output.supports_animation() => {
            MagickConvertJob::from_default_arguments(
                input_path,
                output_path,
                None,
                false,
                false,
                default_arguments,
            )
        }
        (input, output) => MagickConvertJob::from_default_arguments(
            input_path,
            output_path,
            None,
            true,
            !input.supports_alpha() && output.supports_alpha(),
            default_arguments,
        ),
    }
}

pub fn wait_for_child(child: &std::sync::Arc<SharedChild>) -> Result<(), std::io::Error> {
    let exit_status = child.wait()?;
    if exit_status.success() {
        Ok(())
    } else {
        let mut stderr = String::new();
        child
            .take_stdout()
            .map(|mut s| s.read_to_string(&mut stderr).ok());
        child
            .take_stderr()
            .map(|mut s| s.read_to_string(&mut stderr).ok());
        Err(std::io::Error::other(stderr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_memavailable_from_proc_meminfo() {
        let sample = "\
MemTotal:        7968696 kB
MemFree:         1607316 kB
MemAvailable:    4034696 kB
Buffers:             556 kB
";
        assert_eq!(parse_meminfo_available_mib(sample), Some(3940));
    }

    #[test]
    fn falls_back_to_half_memtotal_when_memavailable_missing() {
        let sample = "MemTotal:        8388608 kB\nMemFree:         1024000 kB\n";
        assert_eq!(parse_meminfo_available_mib(sample), Some(4096));
    }

    #[test]
    fn bounds_concurrency_on_8gb_8core_machine_for_batch() {
        // 8 CPUs, ~3.3 GiB available RAM, 30 conversion jobs
        let limits = ResourceLimits::from_system(30, 8, Some(3300));
        assert_eq!(limits.max_concurrent_jobs, 2);
        assert_eq!(limits.threads_per_job, 3);
        assert_eq!(limits.memory_limit_mib, Some(990));
        assert!(limits.max_concurrent_jobs * limits.threads_per_job < 8);
    }

    #[test]
    fn drops_to_single_job_under_low_memory() {
        // 8 CPUs, 2 GiB available RAM, 20 conversion jobs
        let limits = ResourceLimits::from_system(20, 8, Some(2048));
        assert_eq!(limits.max_concurrent_jobs, 1);
        assert_eq!(limits.threads_per_job, 6);
        assert_eq!(limits.memory_limit_mib, Some(1228));
    }

    #[test]
    fn gives_full_cpu_budget_to_single_job() {
        let limits = ResourceLimits::from_system(1, 8, Some(12_000));
        assert_eq!(limits.max_concurrent_jobs, 6);
        assert_eq!(limits.threads_per_job, 6);
    }

    #[test]
    fn scales_up_on_high_memory_workstation_without_saturating_all_cores() {
        let limits = ResourceLimits::from_system(40, 16, Some(24_000));
        assert_eq!(limits.max_concurrent_jobs, 12);
        assert_eq!(limits.threads_per_job, 1);
        assert!(limits.max_concurrent_jobs * limits.threads_per_job < 16);
    }

    #[test]
    fn handles_single_and_dual_core_systems() {
        let single = ResourceLimits::from_system(10, 1, Some(4096));
        assert_eq!(single.max_concurrent_jobs, 1);
        assert_eq!(single.threads_per_job, 1);

        let dual = ResourceLimits::from_system(10, 2, Some(4096));
        assert_eq!(dual.max_concurrent_jobs, 1);
        assert_eq!(dual.threads_per_job, 1);
    }

    #[test]
    fn identify_uses_single_thread_and_bounded_parallelism() {
        let limits = ResourceLimits::for_identify(30, 8, Some(3900));
        assert_eq!(limits.max_concurrent_jobs, 6);
        assert_eq!(limits.threads_per_job, 1);
    }

    #[test]
    fn skips_resize_flag_at_100_percent() {
        assert_eq!(
            ResizeArgument::Percentage {
                width: 100,
                height: 100,
            }
            .get_argument(),
            Vec::<String>::new()
        );
        assert_eq!(
            ResizeArgument::Percentage {
                width: 50,
                height: 50,
            }
            .get_argument(),
            vec!["-resize", "50%x50%"]
        );
    }

    #[test]
    fn rejects_pixel_like_numbers_in_percentage_mode() {
        assert_eq!(
            validate_resize_inputs(true, "2920", "2711", &[(2266, 3200)]),
            Err(ResizeValidationError::PercentageTooLarge)
        );
        assert_eq!(
            validate_resize_inputs(true, "0", "100", &[(2266, 3200)]),
            Err(ResizeValidationError::InvalidPercentage)
        );
        assert_eq!(
            validate_resize_inputs(true, "", "100", &[(2266, 3200)]),
            Err(ResizeValidationError::InvalidPercentage)
        );
    }

    #[test]
    fn rejects_excessive_output_megapixels() {
        // 500% of 5000x5000 = 25000x25000 = 625 MP > 250 MP limit
        assert_eq!(
            validate_resize_inputs(true, "500", "500", &[(5000, 5000)]),
            Err(ResizeValidationError::OutputTooLarge)
        );
        // 16384x16384 = 268 MP > 250 MP limit
        assert_eq!(
            validate_resize_inputs(false, "16384", "16384", &[]),
            Err(ResizeValidationError::OutputTooLarge)
        );
        // Valid 2920x2711 in pixel mode succeeds
        assert_eq!(
            validate_resize_inputs(false, "2920", "2711", &[(2266, 3200)]),
            Ok(ResizeArgument::ExactPixels {
                width: 2920,
                height: 2711,
            })
        );
    }
}
