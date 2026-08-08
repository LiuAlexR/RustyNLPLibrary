use std::{env, fs, path::Path};

/// Reads the file name matching file_path within the 'texts' directory, found in the root of the Github repo. Used for development purposes.
/// Arguments: 
pub fn retrieve_source(file_path: &str) -> String {
    // 'path' becomes the root of the Rust file; e.g., path = path/to/project/rusty_lib
    let path = match env::current_dir() {
        Ok(x) => x,
        Err(_) => {
            panic!("Error reading current path. Something went wrong.");
        },
    };
    // project_root is the Git source, it is also where we navigate to the texts
    let path_parent: &Path = match path.parent() {
        Some(found_path) => found_path,
        None => {
            panic!("Error navigating to project root.");
        },
    };
    // nagivates and selects the file
    let mut text_path = path_parent.to_path_buf();
    text_path.push("texts");
    text_path.push(file_path);
    // reads the bytes
    let file_bytes = match fs::read(text_path) {
        Ok(contents) => contents,
        Err(_) => {
            panic!("Error. File does not exist");
        },
    };
    // converts the byte vec to a String
    let file_contents = match String::from_utf8(file_bytes) {
        Ok(contents) => contents,
        Err(_) => {
            panic!("Error. Unable to convert file to String.");
        },
    };
    // returns the String
    file_contents
}