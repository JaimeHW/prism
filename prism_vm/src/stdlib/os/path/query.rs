//! Path query operations (exists, isfile, isdir, islink).

use std::fs;
use std::path::Path;

/// Check if path exists.
#[inline]
pub fn exists<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().exists()
}

/// Check if path is a file.
#[inline]
pub fn isfile<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().is_file()
}

/// Check if path is a directory.
#[inline]
pub fn isdir<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().is_dir()
}

/// Check if path is a symbolic link.
#[inline]
pub fn islink<P: AsRef<Path>>(path: P) -> bool {
    fs::symlink_metadata(path.as_ref())
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false)
}

/// Check if path is absolute.
#[inline]
pub fn isabs<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().is_absolute()
}

/// Get file size, or None if not accessible.
#[inline]
pub fn getsize<P: AsRef<Path>>(path: P) -> Option<u64> {
    fs::metadata(path.as_ref()).ok().map(|m| m.len())
}

/// Check if path exists, including broken symbolic links.
///
/// Unlike `exists()`, this returns `true` for broken symlinks
/// (where the link itself exists but its target does not).
///
/// Equivalent to Python's `os.path.lexists()`.
#[inline]
pub fn lexists<P: AsRef<Path>>(path: P) -> bool {
    fs::symlink_metadata(path.as_ref()).is_ok()
}

/// Check if path is a mount point.
///
/// On Unix, checks if the path and its parent are on different devices,
/// or if the path is the filesystem root.
///
/// On Windows, checks if the path is a drive root (e.g., `C:\`).
///
/// Equivalent to Python's `os.path.ismount()`.
pub fn ismount<P: AsRef<Path>>(path: P) -> bool {
    ismount_impl(path.as_ref())
}

#[cfg(windows)]
fn ismount_impl(path: &Path) -> bool {
    // On Windows, a mount point is a drive root like "C:\"
    let s = path.to_string_lossy();
    // Drive root: exactly "X:\" or "X:/"
    if s.len() == 3 {
        let bytes = s.as_bytes();
        return bytes[0].is_ascii_alphabetic()
            && bytes[1] == b':'
            && (bytes[2] == b'\\' || bytes[2] == b'/');
    }
    // UNC root: \\server\share
    if s.starts_with("\\\\") {
        let rest = &s[2..];
        // Count backslash separators — a UNC root has exactly server\share
        let parts: Vec<&str> = rest.splitn(3, '\\').collect();
        return parts.len() == 2 || (parts.len() == 3 && parts[2].is_empty());
    }
    false
}

#[cfg(not(windows))]
fn ismount_impl(path: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;

    // Root is always a mount point
    if path == Path::new("/") {
        return true;
    }

    // Compare device IDs of path and its parent
    let Ok(meta) = fs::metadata(path) else {
        return false;
    };
    let Some(parent) = path.parent() else {
        return false;
    };
    let Ok(parent_meta) = fs::metadata(parent) else {
        return false;
    };

    // Different device = mount point
    if meta.dev() != parent_meta.dev() {
        return true;
    }
    // Same inode as parent = root of filesystem
    meta.ino() == parent_meta.ino()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_exists_curdir() {
        assert!(exists("."));
    }

    #[test]
    fn test_exists_nonexistent() {
        assert!(!exists("/nonexistent_12345"));
    }

    #[test]
    fn test_isfile_on_dir() {
        assert!(!isfile(env::temp_dir()));
    }

    #[test]
    fn test_isdir_temp() {
        assert!(isdir(env::temp_dir()));
    }

    #[test]
    fn test_isdir_on_file() {
        // Current exe is a file
        let exe = env::current_exe().unwrap();
        assert!(!isdir(exe));
    }

    #[test]
    fn test_isabs_absolute() {
        #[cfg(windows)]
        assert!(isabs("C:\\Windows"));
        #[cfg(not(windows))]
        assert!(isabs("/usr"));
    }

    #[test]
    fn test_isabs_relative() {
        assert!(!isabs("foo/bar"));
    }

    #[test]
    fn test_getsize_some() {
        let exe = env::current_exe().unwrap();
        let size = getsize(exe);
        assert!(size.is_some());
        assert!(size.unwrap() > 0);
    }

    #[test]
    fn test_getsize_none() {
        assert!(getsize("/nonexistent_12345").is_none());
    }

    // =========================================================================
    // lexists Tests
    // =========================================================================

    #[test]
    fn test_lexists_curdir() {
        assert!(lexists("."));
    }

    #[test]
    fn test_lexists_nonexistent() {
        assert!(!lexists("/nonexistent_12345"));
    }

    #[test]
    fn test_lexists_file() {
        let exe = env::current_exe().unwrap();
        assert!(lexists(&exe));
    }

    #[test]
    fn test_lexists_directory() {
        assert!(lexists(env::temp_dir()));
    }

    // =========================================================================
    // ismount Tests
    // =========================================================================

    #[cfg(windows)]
    #[test]
    fn test_ismount_drive_root() {
        assert!(ismount("C:\\"));
    }

    #[cfg(windows)]
    #[test]
    fn test_ismount_non_root() {
        assert!(!ismount("C:\\Windows"));
    }

    #[cfg(not(windows))]
    #[test]
    fn test_ismount_root() {
        assert!(ismount("/"));
    }

    #[cfg(not(windows))]
    #[test]
    fn test_ismount_non_root() {
        assert!(!ismount("/tmp") || std::path::Path::new("/tmp").exists());
    }

    #[test]
    fn test_ismount_nonexistent() {
        assert!(!ismount("/nonexistent_12345"));
    }
}
