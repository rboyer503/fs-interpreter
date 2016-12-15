FSI Data Collection Tool
========================

Overview
--------
This utility is designed to simplify collection of data samples used for training the Convolutional Neural Nets (CNNs) that support the core symbol classification used for fingerspelling interpretation.  It was written entirely in C++ using Visual Studio 2010.  A full C++ implementation of the hand location, tracking, and feature extraction pipeline is included as described in the master README.md file.  Various keys are defined to trigger actions such as capturing a burst of snapshots; toggling modes; compiling raw data and label files from a collection of snapshots; loading, displaying, and navigating an existing raw data file; etc...


Dependencies
------------
- OpenCV 2.4.11 (Core, HighGUI, ImgProc, and ObjDetect)
  - Minor patch required for restoring auto-exposure - updated file included at ./OpenCVPatch/cap_dshow.cpp.
- Windows Multimedia API (WinMM.lib)
- Boost 1.55 (Only for posix_time - could be removed without much effort.)
- Visual C++ 2010 SP1 x64 runtime components
  - For pre-built binary only (available at ./x64/Release/FSIDataCollectTool.exe).
  - Redistributable can be found at https://www.microsoft.com/en-us/download/details.aspx?id=26999.


Usage
-----
    FSIDataCollectTool <device ID>
        <device ID>: ID of video device (0-9)

Example usage

1. Start the FSI Data Collect Tool.
2. Hold the 'A' symbol (basic closed hand facing forward with thumb on side) roughly 2 to 6 feet from webcam until software locates hand and enters the TRACK_HAND state.
3. Begin collecting samples.
  1. While holding the 'A' symbol, hit 'a' to capture 10 snapshots to .\TempData\a.
  2. While holding the 'B' symbol, hit 'B' to capture 20 snapshots to .\TempData\b.
  3. Hit '!' to toggle to single image capture mode.
  4. While holding the sentinel symbol, hit 'z' to capture a single snapshot to .\TempData\z.
  5. Hit '!' again to return to burst image capture mode.
4. Move all subdirectories under .\TempData into a new subdirectory under .\data (for example, .\data\MyData).
5. Capture samples involving motion.
  1. Hit '@' to toggle to J/Z capture mode.
  2. Hit 'j' and perform the 'J' gesture to capture the clip.
  3. Hit 'z' and perform the 'Z' gesture to capture another clip.
  4. Hit 'q' and do neither 'J' nor 'Z' to capture a "neither" clip.
  5. Hit '@' to return to normal capture mode.
6. Move all subdirectories under .\TempData into a new subdirectory under .\clipData (for example, .\clipData\MyClipData).
7. Hit '1' to generate the raw data and label files for the data collected under .\data.
8. Hit '2' to load the data.raw file.
9. Hit '>' to navigate to the first sample; sample is displayed in the Data Collection Viewer window.
10. When done, hit the Escape key to exit.

Miscellaneous usage notes:
- Timing the capture of a burst/clip of snapshots can be challenging.  To make this process easier, the software plays 4 beep sounds at regular intervals.  Immediately after the 3rd beep, we begin capturing snapshots.  All snapshots are collected by the 4th beep: (beep ... beep ... beep ... `<capture snapshots>` ... beep).  It is recommended to practice several times before attempting to collect real samples.


Main Menu
---------
Press the desired key to trigger various actions as defined in the menu.

    ESC:    Exit
    SPACE:  Pause
    ?:      Display this menu
    !:      Toggle single image capture mode
    @:      Toggle J/Z capture mode
    [a..z]: Capture 10 snapshots
    [A..Z]: Capture 20 snapshots
    1:      Generate data and label files
    2:      Load data file
    3:      Split data and label files
    4:      Calculate data file statistics
    </>:    Move back/forward one image
    [/]:    Move back/forward one letter
    {/}:    Move back/forward one collection
    +/-:    Increase/decrease camera exposure


Actions
-------
[Exit]
Terminate the program.

[Pause]
Toggle the pause state.  While paused, all image processing is suppressed, locking the display at the last processed image.

[Toggle single image capture mode]
In single image capture mode, capture actions capture individual snapshots instead of bursts.  In addition, letter/collection navigation actions are adjusted to move by number of samples per capture rather than number of samples per letter.

[Toggle J/Z capture mode]
In J/Z capture mode, short clips of 8 snapshots are captured instead of individual or "bursts" of snapshots.  This is to support data collection for letters that involve motion: the letters J and Z.  These 8-frame clips represent a single data sample.  In J/Z capture mode, the 'j', 'z', and 'q' keys trigger captures for samples of the 'J' gesture, 'Z' gesture, and neither respectively.

[Capture 10/20 snapshots]
Capture a snapshot or burst of snapshots of the letter of the corresponding key press.  For example, pressing 'b' will trigger a capture of 10 snapshots which will be stored under the .\TempData\b subdirectory (assuming not in single capture mode and not in J/Z capture mode).

[Generate data and label files]
If in normal mode, process all data samples under the .\data directory to generate two raw files: data.raw and labels.raw.  For J/Z mode, process all clip samples under the .\clipData directory to generate two raw files: clipData.raw and clipLabels.raw.  The data file stores the actual features while the labels file stores the corresponding labels.  Directory processing is recursive, so unlimited collections of samples can be stored under .\data or .\clipData.  Typically, a user will collect data for all letters in a session and then move the samples from .\TempData to a sub-directory under .\data and/or .\clipData beforing generating the raw data and label files.
Note that this process will create NUM_SAMPLES_PER_CAPTURE (see Globals.h) samples in the output raw files per each sample.  (This involves some typical "data multiplication" processes such as randomly adjusting the scale, angle, and constrast.)

[Load data file]
If in normal mode, open data.raw for display/navigation in the Data Collection Viewer window.  For J/Z mode, open clipData.raw for similar purposes.  The corresponding file must exist in the local directory for proper behavior.  Use the </>, [/], {/} keys to navigate after loading the file.

[Split data and label files]
If in normal mode, read data.raw and labels.raw and split them into NUM_SPLIT_FILES (see Globals.h) files.  Used to keep file size manageable; subsequent training of CNNs can cycle through each file in turn (generally making several passes through each file).  Samples are divvied up one-by-one to the files to avoid skewed distributions.

[Calculate data file statistics]
Calculate YCrCb statistics on all samples in the .\data directory.  This was used primarily to tune logic that discards false positives from the cascade classifier during the hand location phase.

[Move back/forward one image]
Used to navigate through individual samples in the raw data file.  Run the [Load data file] action first.

[Move back/forward one letter]
Navigate through samples, jumping to the next or previous letter.  The number of samples skipped is based on NUM_SAMPLES_PER_CAPTURE and NUM_CAPTURES_PER_LETTER (or CLIP_LENGTH for J/Z mode) as defined in Globals.h.

[Move back/forward one collection]
Navigate through samples, jumping to the next or previous collection.  This is similar to navigation by letter, except the jump is multiplied by the number of letters in the normal data collection (25: all letters except 'j' and 'z', plus the sentinel).
Note: Avoid this for J/Z mode as it is not currently tuned for proper navigation of clip data.

[Increase/decrease camera exposure]
Manually adjust the exposure.  This can be useful in cases where webcam auto-adjust is not properly adjusting to the lighting conditions or to deliberately generate data samples with varied exposures to improve generalization to lighting conditions.