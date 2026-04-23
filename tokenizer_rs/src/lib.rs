/*!
 * kmamba_tokenizer - Hybrid Tokenizer FFI for k-mamba
 */

use std::ffi::{CStr, CString, c_char};
use std::os::raw::c_uint;
use std::slice;
use std::sync::OnceLock;
use tiktoken_rs::{cl100k_base_singleton, CoreBPE};
use parking_lot::RwLock;

#[derive(Debug, Clone, Copy)]
pub enum TokenizerType {
    Bytes32K,      // Raw bytes 0-255, vocab_size = 32768 (padded)
    Tiktoken100K,  // cl100k_base, vocab_size = 100277
}

struct GlobalTokenizer {
    t_type: TokenizerType,
    bpe: Option<std::sync::Arc<parking_lot::Mutex<CoreBPE>>>,
}

static TOKENIZER: OnceLock<RwLock<GlobalTokenizer>> = OnceLock::new();

fn get_tokenizer() -> &'static RwLock<GlobalTokenizer> {
    TOKENIZER.get_or_init(|| {
        RwLock::new(GlobalTokenizer {
            t_type: TokenizerType::Bytes32K,
            bpe: None,
        })
    })
}

/// Initialize the tokenizer with a specific type
/// type_str: "bytes" or "cl100k"
#[no_mangle]
pub unsafe extern "C" fn kmamba_tokenizer_init(type_str: *const c_char) -> i32 {
    if type_str.is_null() { return -1; }
    let c_str = CStr::from_ptr(type_str).to_str().unwrap_or("bytes");
    
    let mut t = get_tokenizer().write();
    match c_str {
        "cl100k" => {
            t.t_type = TokenizerType::Tiktoken100K;
            t.bpe = Some(cl100k_base_singleton());
            0
        },
        "bytes" | _ => {
            t.t_type = TokenizerType::Bytes32K;
            t.bpe = None;
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn kmamba_encode(text: *const c_char, out_len: *mut usize) -> *mut c_uint {
    if text.is_null() || out_len.is_null() { return std::ptr::null_mut(); }
    
    let c_str = match CStr::from_ptr(text).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    
    let t = get_tokenizer().read();
    let tokens: Vec<c_uint> = match t.t_type {
        TokenizerType::Bytes32K => {
            // Byte-level: each byte is a token (0-255)
            c_str.as_bytes().iter().map(|&b| b as c_uint).collect()
        },
        TokenizerType::Tiktoken100K => {
            if let Some(ref bpe) = t.bpe {
                bpe.lock().encode_with_special_tokens(c_str)
                    .into_iter().map(|v| v as c_uint).collect()
            } else {
                return std::ptr::null_mut();
            }
        }
    };
    
    let len = tokens.len();
    let layout = std::alloc::Layout::array::<c_uint>(len).unwrap();
    let ptr = std::alloc::alloc(layout) as *mut c_uint;
    if ptr.is_null() { return std::ptr::null_mut(); }
    
    std::ptr::copy_nonoverlapping(tokens.as_ptr(), ptr, len);
    *out_len = len;
    ptr
}

#[no_mangle]
pub unsafe extern "C" fn kmamba_decode(tokens: *const c_uint, len: usize) -> *mut c_char {
    if tokens.is_null() || len == 0 { return std::ptr::null_mut(); }
    let token_slice = slice::from_raw_parts(tokens, len);
    
    let t = get_tokenizer().read();
    let text = match t.t_type {
        TokenizerType::Bytes32K => {
            let bytes: Vec<u8> = token_slice.iter().map(|&v| (v & 0xFF) as u8).collect();
            String::from_utf8_lossy(&bytes).into_owned()
        },
        TokenizerType::Tiktoken100K => {
            if let Some(ref bpe) = t.bpe {
                match bpe.lock().decode(token_slice.iter().map(|&v| v as u32).collect()) {
                    Ok(s) => s,
                    Err(_) => return std::ptr::null_mut(),
                }
            } else {
                return std::ptr::null_mut();
            }
        }
    };
    
    match CString::new(text) {
        Ok(cstr) => cstr.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn kmamba_free_tokens(ptr: *mut c_uint, len: usize) {
    if !ptr.is_null() && len > 0 {
        let layout = std::alloc::Layout::array::<c_uint>(len).unwrap();
        std::alloc::dealloc(ptr as *mut u8, layout);
    }
}

#[no_mangle]
pub unsafe extern "C" fn kmamba_free_string(ptr: *mut c_char) {
    if !ptr.is_null() { let _ = CString::from_raw(ptr); }
}

#[no_mangle]
pub extern "C" fn kmamba_vocab_size() -> usize {
    let t = get_tokenizer().read();
    match t.t_type {
        TokenizerType::Bytes32K => 32768,
        TokenizerType::Tiktoken100K => 100277,
    }
}
