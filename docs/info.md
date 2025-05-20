- **Import the logger:**  
    Use the `logger` (with rich formatting) from `image_caption.config`. Use `logger.setLevel(logging.WARNING)` to set 
    the logging level (in this example, to level=WARNING).

- **Log directory:**  
    By default, a `logs/` directory is created in the parent directory of the script. You can change this location in `image_caption/config.py`.

- **Enhanced logging:**  
    The `rich` library is used to improve log readability. Use `logger.info()` or `logger.error()` to write output logs to these directories.

---

### Example Log Output from `main.py`

The following is a sample of the terminal output (such as model characteristics and behavior) that is saved to the log file:

```
INFO 2025-05-20 07:24:14,239 [root:main.py:<module>:27]
Total images: 8091

INFO 2025-05-20 07:24:14,255 [root:main.py:<module>:45]
Vocabulary Size: 814

INFO 2025-05-20 07:24:14,258 [root:main.py:<module>:49]
Maximum caption length: 27

INFO 2025-05-20 07:24:14,689 [root:main.py:<module>:60]
Feature vector shape: (1, 256)

INFO 2025-05-20 07:24:14,799 [root:main.py:<module>:62]
Feature vector shape: (1, 256)

INFO 2025-05-20 07:24:23,491 [root:main.py:<module>:78]
Extracted features for 100 images

INFO 2025-05-20 07:24:58,581 [root:main.py:<module>:134]
Generated caption: startseq dog is and dog in in endseq
```

