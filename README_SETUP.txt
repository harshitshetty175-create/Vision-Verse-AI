VISION VERSE - SETUP AND RUN GUIDE
=================================

PROJECT CONTENTS
----------------
01_webcam_test.py       Basic webcam test
02_hand_tracking.py     MediaPipe 21-landmark hand tracking test (updated)
pinch_pop.py            Main interactive AI Photo Booth application
hand_landmarker.task    MediaPipe hand model
captures/               Saved photos, videos and photo strips
requirements.txt        Python dependencies

IMPORTANT
---------
The bundled .venv folder from the original ZIP was intentionally not copied.
A virtual environment is specific to the computer/Python installation that
created it. Create a fresh environment on your laptop instead.

SETUP (Windows / VS Code)
-------------------------
1. Extract this ZIP completely.
2. Open the INNER "Vision Verse" folder in VS Code.
   It is the folder containing pinch_pop.py and hand_landmarker.task.
3. Open Terminal > New Terminal.
4. Install dependencies:

   py -3.13 -m pip install --upgrade pip
   py -3.13 -m pip install -r requirements.txt

5. Test the webcam:

   py -3.13 01_webcam_test.py

   Press Q to close it.

6. Test MediaPipe hand landmarks:

   py -3.13 02_hand_tracking.py

   Show your hand to the webcam. You should see the 21-point hand skeleton.
   Touch thumb and index finger to see PINCH status. Press Q to close.

7. Run the MAIN PHOTO BOOTH application:

   py -3.13 pinch_pop.py

MAIN APP CONTROLS
-----------------
- Pinch with the right hand to draw.
- Pinch with the left hand to erase.
- Pinch over a colour or filter to select it.
- Pinch the round photo button for a 3-second photo countdown.
- Select BOOTH for four timed photos and a photo strip.
- Z = undo
- C = clear drawing
- 1 to 5 = choose filters
- P = take a photo
- B = start/cancel four-photo booth
- V = start/stop video recording
- S = swap left/right hand roles if they are reversed
- K = show/hide hand skeleton
- H = show/hide help
- Q or ESC = quit

OUTPUTS
-------
Photos, videos and four-photo strips are saved in the captures folder.

TROUBLESHOOTING
---------------
- If the webcam does not open, close Teams/Zoom/Camera and retry.
- Allow camera access in Windows Settings > Privacy & security > Camera.
- If MediaPipe cannot find the model, confirm hand_landmarker.task is in the
  same folder as pinch_pop.py and 02_hand_tracking.py.
- Run commands from the inner Vision Verse folder, not its parent folder.
- Do not install opencv-python-headless together with opencv-python.
