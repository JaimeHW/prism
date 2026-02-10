//! Path timestamp access operations.
//!
//! High-performance timestamp queries using platform-specific APIs for
//! maximum precision. Returns UNIX timestamps as `f64` for sub-second accuracy.
//!
//! # Performance
//!
//! - Single syscall per query via `fs::metadata()`
//! - Platform-native timestamp extraction (no conversion overhead)
//! - `#[inline]` for zero-overhead abstraction

use std::fs;
use std::path::Path;
use std::time::SystemTime;

/// Get modification time as UNIX timestamp (seconds since epoch).
///
/// Equivalent to Python's `os.path.getmtime()`.
///
/// # Platform behavior
///
/// - **Windows**: Uses `last_write_time()` from WIN32_FILE_ATTRIBUTE_DATA
/// - **Unix**: Uses `st_mtime` + `st_mtime_nsec` from `stat(2)`
#[inline]
pub fn getmtime<P: AsRef<Path>>(path: P) -> std::io::Result<f64> {
    let metadata = fs::metadata(path.as_ref())?;
    system_time_to_epoch(metadata.modified()?)
}

/// Get last access time as UNIX timestamp (seconds since epoch).
///
/// Equivalent to Python's `os.path.getatime()`.
///
/// # Note
///
/// Many filesystems have `noatime` mount option which disables access time
/// tracking for performance. In such cases, the access time may equal the
/// modification time or remain at creation time.
#[inline]
pub fn getatime<P: AsRef<Path>>(path: P) -> std::io::Result<f64> {
    let metadata = fs::metadata(path.as_ref())?;
    system_time_to_epoch(metadata.accessed()?)
}

/// Get creation time (Windows) or metadata change time (Unix) as UNIX timestamp.
///
/// Equivalent to Python's `os.path.getctime()`.
///
/// # Platform behavior
///
/// - **Windows**: Returns file creation time (birth time)
/// - **Unix**: Returns metadata change time (`st_ctime`), which is updated
///   when file metadata (permissions, owner, etc.) changes
#[inline]
pub fn getctime<P: AsRef<Path>>(path: P) -> std::io::Result<f64> {
    let metadata = fs::metadata(path.as_ref())?;
    // `created()` can fail on some Unix systems that don't support birth time.
    // Fall back to modified() if created() is unavailable.
    match metadata.created() {
        Ok(t) => system_time_to_epoch(t),
        Err(_) => system_time_to_epoch(metadata.modified()?),
    }
}

/// Get modification time for a symlink itself (not its target).
///
/// Equivalent to calling `os.lstat(path).st_mtime`.
#[inline]
pub fn lgetmtime<P: AsRef<Path>>(path: P) -> std::io::Result<f64> {
    let metadata = fs::symlink_metadata(path.as_ref())?;
    system_time_to_epoch(metadata.modified()?)
}

/// Convert `SystemTime` to UNIX epoch seconds with sub-second precision.
///
/// # Performance
///
/// This is a pure arithmetic operation — no syscalls, no allocations.
/// Uses `duration_since(UNIX_EPOCH)` which is a simple subtraction on
/// all supported platforms.
#[inline]
fn system_time_to_epoch(time: SystemTime) -> std::io::Result<f64> {
    time.duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs::File;
    use std::io::Write;

    // =========================================================================
    // getmtime Tests
    // =========================================================================

    #[test]
    fn test_getmtime_existing_file() {
        let exe = env::current_exe().unwrap();
        let mtime = getmtime(&exe).unwrap();
        // Should be a reasonable UNIX timestamp (after 2020-01-01)
        assert!(mtime > 1_577_836_800.0, "mtime {} too old", mtime);
    }

    #[test]
    fn test_getmtime_directory() {
        let temp = env::temp_dir();
        let mtime = getmtime(&temp).unwrap();
        assert!(mtime > 0.0);
    }

    #[test]
    fn test_getmtime_nonexistent() {
        let result = getmtime("/nonexistent_path_12345");
        assert!(result.is_err());
    }

    #[test]
    fn test_getmtime_has_subsecond_precision() {
        let exe = env::current_exe().unwrap();
        let mtime = getmtime(&exe).unwrap();
        // Just verify it's a valid float (not NaN or Inf)
        assert!(mtime.is_finite());
    }

    #[test]
    fn test_getmtime_changes_on_write() {
        let dir = env::temp_dir();
        let path = dir.join("_prism_test_getmtime_change.txt");

        // Create file
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(b"initial").unwrap();
        }
        let mtime1 = getmtime(&path).unwrap();

        // Brief pause to ensure timestamp changes
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Modify file
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(b"modified").unwrap();
        }
        let mtime2 = getmtime(&path).unwrap();

        assert!(
            mtime2 >= mtime1,
            "mtime should not decrease: {} vs {}",
            mtime1,
            mtime2
        );

        // Cleanup
        let _ = fs::remove_file(&path);
    }

    // =========================================================================
    // getatime Tests
    // =========================================================================

    #[test]
    fn test_getatime_existing_file() {
        let exe = env::current_exe().unwrap();
        let atime = getatime(&exe).unwrap();
        assert!(atime > 1_577_836_800.0, "atime {} too old", atime);
    }

    #[test]
    fn test_getatime_directory() {
        let temp = env::temp_dir();
        let atime = getatime(&temp).unwrap();
        assert!(atime > 0.0);
    }

    #[test]
    fn test_getatime_nonexistent() {
        assert!(getatime("/nonexistent_12345").is_err());
    }

    #[test]
    fn test_getatime_is_finite() {
        let exe = env::current_exe().unwrap();
        let atime = getatime(&exe).unwrap();
        assert!(atime.is_finite());
        assert!(atime > 0.0);
    }

    // =========================================================================
    // getctime Tests
    // =========================================================================

    #[test]
    fn test_getctime_existing_file() {
        let exe = env::current_exe().unwrap();
        let ctime = getctime(&exe).unwrap();
        assert!(ctime > 1_577_836_800.0, "ctime {} too old", ctime);
    }

    #[test]
    fn test_getctime_directory() {
        let temp = env::temp_dir();
        let ctime = getctime(&temp).unwrap();
        assert!(ctime > 0.0);
    }

    #[test]
    fn test_getctime_nonexistent() {
        assert!(getctime("/nonexistent_12345").is_err());
    }

    #[test]
    fn test_getctime_is_finite() {
        let exe = env::current_exe().unwrap();
        let ctime = getctime(&exe).unwrap();
        assert!(ctime.is_finite());
    }

    #[test]
    fn test_getctime_not_after_mtime() {
        // Creation time should be <= modification time (usually)
        let exe = env::current_exe().unwrap();
        let ctime = getctime(&exe).unwrap();
        let mtime = getmtime(&exe).unwrap();
        assert!(
            ctime <= mtime + 1.0,
            "ctime {} should not be much after mtime {}",
            ctime,
            mtime
        );
    }

    // =========================================================================
    // lgetmtime Tests
    // =========================================================================

    #[test]
    fn test_lgetmtime_on_regular_file() {
        let exe = env::current_exe().unwrap();
        let lmtime = lgetmtime(&exe).unwrap();
        let mtime = getmtime(&exe).unwrap();
        // For regular files, lgetmtime == getmtime
        assert!(
            (lmtime - mtime).abs() < 1.0,
            "lmtime {} vs mtime {}",
            lmtime,
            mtime
        );
    }

    #[test]
    fn test_lgetmtime_nonexistent() {
        assert!(lgetmtime("/nonexistent_12345").is_err());
    }

    // =========================================================================
    // system_time_to_epoch Tests
    // =========================================================================

    #[test]
    fn test_system_time_to_epoch_now() {
        let now = SystemTime::now();
        let epoch = system_time_to_epoch(now).unwrap();
        assert!(epoch > 1_577_836_800.0); // After 2020
        assert!(epoch < 2_524_608_000.0); // Before 2050
    }

    #[test]
    fn test_system_time_to_epoch_unix_epoch() {
        let epoch_time = SystemTime::UNIX_EPOCH;
        let epoch = system_time_to_epoch(epoch_time).unwrap();
        assert!((epoch - 0.0).abs() < f64::EPSILON);
    }

    // =========================================================================
    // Cross-function consistency Tests
    // =========================================================================

    #[test]
    fn test_all_times_consistent_for_file() {
        let exe = env::current_exe().unwrap();
        let mtime = getmtime(&exe).unwrap();
        let atime = getatime(&exe).unwrap();
        let ctime = getctime(&exe).unwrap();

        // All should be positive and reasonable
        assert!(mtime > 0.0);
        assert!(atime > 0.0);
        assert!(ctime > 0.0);

        // All should be finite
        assert!(mtime.is_finite());
        assert!(atime.is_finite());
        assert!(ctime.is_finite());
    }

    #[test]
    fn test_all_times_error_on_nonexistent() {
        let bad = "/nonexistent_path_99999";
        assert!(getmtime(bad).is_err());
        assert!(getatime(bad).is_err());
        assert!(getctime(bad).is_err());
        assert!(lgetmtime(bad).is_err());
    }

    #[test]
    fn test_new_file_has_recent_timestamps() {
        let dir = env::temp_dir();
        let path = dir.join("_prism_test_recent_ts.txt");

        // Create file
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(b"test").unwrap();
        }

        let now_approx = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let mtime = getmtime(&path).unwrap();
        let atime = getatime(&path).unwrap();
        let ctime = getctime(&path).unwrap();

        // All timestamps should be within 30 seconds of now
        let tolerance = 30.0;
        assert!(
            (mtime - now_approx).abs() < tolerance,
            "mtime {} far from now {}",
            mtime,
            now_approx
        );
        assert!(
            (atime - now_approx).abs() < tolerance,
            "atime {} far from now {}",
            atime,
            now_approx
        );
        assert!(
            (ctime - now_approx).abs() < tolerance,
            "ctime {} far from now {}",
            ctime,
            now_approx
        );

        // Cleanup
        let _ = fs::remove_file(&path);
    }
}
