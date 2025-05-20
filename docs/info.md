- `logger` with rich can be imported from `image_caption.config`. 
- A logs/ directory will be created in the parent directory of the script where it is used, but you may change it in image_caption/config.py. 
- `rich` is used to enhance logs. Use logger.info() or logger.error() to write to a running output log in these directories. 

**BELOW IS EXAMPLE INFO OUTPUT FROM `main.py`**
- You can see that the terminal output, such as model characteristics and behavior, is output to this file. 

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

