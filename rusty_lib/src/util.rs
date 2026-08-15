use std::{env, fs, io, path::Path};

/// Reads the file name matching file_path within the 'texts' directory, found in the root of the Github repo. Used for development purposes.
/// Arguments: the file name, e.g., "orwell_1984.txt"
/// Returns the contents of the file, as a String
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

/// Writes a String to a file within the 'storage' folder in the root. Will override if the file already exists, and will create the file if it does not.
pub fn write_to_storage(file_path: &str, contents: &String) -> io::Result<()> {
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
    text_path.push("storage");
    text_path.push(file_path);
    // Writes the file, and returns the Result of the operation
    fs::write(text_path, contents)
}
/// Joins a vector to a String
pub fn vec_to_string(delimiter: &str, the_vec: &Vec<String>) -> String {
    the_vec.join(delimiter)
}

pub fn string_to_vec(delimiter: &str, the_string: &String) -> Vec<String> {
    let mut the_vec: Vec<String> = the_string.split(delimiter).map(String::from).collect();
    the_vec.retain(|x| x != "");
    the_vec
}
pub fn read_stored_token(file_path: &str) -> Vec<String> {
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
    text_path.push("storage");
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
    string_to_vec("|", &file_contents)
}