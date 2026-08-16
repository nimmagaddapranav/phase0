Extract frame: ffmpeg -i IMG_3524.mp4 -vf "select=eq(n\,100)" -vframes 1 frame.jpg
Download frame.jpg, annotate in the Court Annotator v5 artifact
Copy the JSON, save as court_calibration.json on EC2
Run: python court_calibration.py --input court_calibration.json --video IMG_3524.mp4




