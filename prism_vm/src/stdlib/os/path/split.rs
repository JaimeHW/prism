//! Path split operations (basename, dirname, splitext).

use std::path::Path;

/// Get the base name (final component) of a path.
#[inline]
pub fn basename<P: AsRef<Path>>(path: P) -> String {
    path.as_ref()
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Get the directory name (all but final component) of a path.
#[inline]
pub fn dirname<P: AsRef<Path>>(path: P) -> String {
    path.as_ref()
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Split path into (root, ext) where ext includes the dot.
pub fn splitext<P: AsRef<Path>>(path: P) -> (String, String) {
    let path = path.as_ref();
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();

    if let Some(dot_pos) = name.rfind('.') {
        if dot_pos > 0 {
            let parent = dirname(&path);
            let stem = &name[..dot_pos];
            let ext = &name[dot_pos..];
            let root = if parent.is_empty() {
                stem.to_string()
            } else {
                format!("{}{}{}", parent, std::path::MAIN_SEPARATOR, stem)
            };
            return (root, ext.to_string());
        }
    }
    (path.to_string_lossy().into_owned(), String::new())
}

/// Split path into (head, tail) where tail is final component.
pub fn split<P: AsRef<Path>>(path: P) -> (String, String) {
    (dirname(&path), basename(&path))
}

/// Split a path into (drive, tail) where drive is a drive letter spec
/// or UNC path prefix, and tail is everything else.
///
/// On Unix, drive is always empty.
///
/// Equivalent to Python's `os.path.splitdrive()`.
///
/// # Examples
///
/// ```ignore
/// // Windows
/// assert_eq!(splitdrive("C:\\foo\\bar"), ("C:".into(), "\\foo\\bar".into()));
/// // Unix
/// assert_eq!(splitdrive("/foo/bar"), ("".into(), "/foo/bar".into()));
/// ```
pub fn splitdrive<P: AsRef<Path>>(path: P) -> (String, String) {
    let s = path.as_ref().to_string_lossy();
    splitdrive_str(&s)
}

/// Internal string-based splitdrive implementation.
fn splitdrive_str(path: &str) -> (String, String) {
    #[cfg(windows)]
    {
        let bytes = path.as_bytes();

        // UNC path: \\server\share
        if bytes.len() >= 2
            && (bytes[0] == b'\\' || bytes[0] == b'/')
            && (bytes[1] == b'\\' || bytes[1] == b'/')
        {
            // Find the end of server\share
            let rest = &path[2..];
            if let Some(sep_pos) = rest.find(|c: char| c == '\\' || c == '/') {
                let after_server = &rest[sep_pos + 1..];
                if let Some(share_end) = after_server.find(|c: char| c == '\\' || c == '/') {
                    let drive_end = 2 + sep_pos + 1 + share_end;
                    return (path[..drive_end].to_string(), path[drive_end..].to_string());
                } else {
                    // \\server\share with no trailing path
                    return (path.to_string(), String::new());
                }
            }
            return (path.to_string(), String::new());
        }

        // Drive letter: X:
        if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
            return (path[..2].to_string(), path[2..].to_string());
        }

        // No drive
        (String::new(), path.to_string())
    }

    #[cfg(not(windows))]
    {
        // On Unix, there's no drive concept
        (String::new(), path.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basename_simple() {
        assert_eq!(basename("/foo/bar.txt"), "bar.txt");
    }

    #[test]
    fn test_basename_no_slash() {
        assert_eq!(basename("file.txt"), "file.txt");
    }

    #[test]
    fn test_basename_trailing_slash() {
        // Behavior depends on OS, just check no panic
        let _ = basename("/foo/bar/");
    }

    #[test]
    fn test_dirname_simple() {
        let d = dirname("/foo/bar.txt");
        assert!(d.contains("foo"));
    }

    #[test]
    fn test_dirname_no_slash() {
        assert_eq!(dirname("file.txt"), "");
    }

    #[test]
    fn test_splitext_with_ext() {
        let (root, ext) = splitext("file.txt");
        assert_eq!(root, "file");
        assert_eq!(ext, ".txt");
    }

    #[test]
    fn test_splitext_no_ext() {
        let (root, ext) = splitext("file");
        assert_eq!(root, "file");
        assert_eq!(ext, "");
    }

    #[test]
    fn test_splitext_hidden() {
        let (root, ext) = splitext(".hidden");
        // .hidden is all extension or no extension depending on interpretation
        assert!(root.contains("hidden") || ext.contains("hidden"));
    }

    #[test]
    fn test_splitext_multi_dot() {
        let (root, ext) = splitext("file.tar.gz");
        assert_eq!(ext, ".gz");
    }

    #[test]
    fn test_split() {
        let (head, tail) = split("/foo/bar.txt");
        assert!(head.contains("foo"));
        assert_eq!(tail, "bar.txt");
    }

    // =========================================================================
    // splitdrive Tests
    // =========================================================================

    #[cfg(windows)]
    #[test]
    fn test_splitdrive_windows_drive() {
        let (drive, tail) = splitdrive("C:\\foo\\bar");
        assert_eq!(drive, "C:");
        assert_eq!(tail, "\\foo\\bar");
    }

    #[cfg(windows)]
    #[test]
    fn test_splitdrive_windows_no_drive() {
        let (drive, tail) = splitdrive("\\foo\\bar");
        // No standard drive letter — treated as relative or UNC start
        assert!(!drive.is_empty() || tail == "\\foo\\bar");
    }

    #[cfg(windows)]
    #[test]
    fn test_splitdrive_windows_drive_only() {
        let (drive, tail) = splitdrive("C:");
        assert_eq!(drive, "C:");
        assert_eq!(tail, "");
    }

    #[cfg(not(windows))]
    #[test]
    fn test_splitdrive_unix_no_drive() {
        let (drive, tail) = splitdrive("/foo/bar");
        assert_eq!(drive, "");
        assert_eq!(tail, "/foo/bar");
    }

    #[cfg(not(windows))]
    #[test]
    fn test_splitdrive_unix_relative() {
        let (drive, tail) = splitdrive("foo/bar");
        assert_eq!(drive, "");
        assert_eq!(tail, "foo/bar");
    }

    #[test]
    fn test_splitdrive_empty() {
        let (drive, tail) = splitdrive("");
        assert_eq!(drive, "");
        assert_eq!(tail, "");
    }

    #[test]
    fn test_splitdrive_single_char() {
        let (drive, tail) = splitdrive("a");
        assert_eq!(drive, "");
        assert_eq!(tail, "a");
    }

    #[test]
    fn test_splitdrive_dot() {
        let (drive, tail) = splitdrive(".");
        assert_eq!(drive, "");
        assert_eq!(tail, ".");
    }
}
