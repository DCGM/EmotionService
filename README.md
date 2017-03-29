# EmotionService
Web service extracting emotions from video


Dependencies
============
Caffe, dlib, OpenCV 3.2, flask, sqlite3 


To start:
=========
  * Checkout submodules `git submodule init; git submodule update`
  * run `./getNets.sh` to download CNNs.
  * Build Openface: 
```
mkdir -p OpenFace/build;
cd OpenFace/build/;
cmake -D CMAKE_BUILD_TYPE=Release  ..;
make -j
```
  * Run server
```
cd service;
./runAll.sh
```
  * Upload video at `localhost:9000`


API
===
* `[POST] /upload` to upload file and get unique identifier
* `[GET] /status/{uuid}` to check processing status
* `[GET] /results/{uuid}` to get per-frame resuls 
* `[GET] /results_agg/{uuid}` to get aggregated result for time intervals

Upload file
===========
Use form at `localhost:9001/` or this curl command:
`curl -v -F "file=@shortTest.mp4" localhost:9001/upload`

Response: 
```
{
  "status": "OK", 
  "uuid": "0e47b9ca11ce494da23fc11b653698ae"
}
```
`Use the `uuid` in other calls to identify this video.

Check processing status
=======================
`[GET] /status/{uuid}`

When [\'data\'][\'status\'] is \"DONE\" in the response, you can access the results.

Response when done:
```
{
  "data": {
    "created": -785.7602241039276, 
    "finished_stages": 6, 
    "lastchange": -750.1767971515656, 
    "original_name": "shortTest.mp4", 
    "stages": [
      {
        "id": 0, 
        "lastchange": -785.5330801010132, 
        "stage": "create_directory", 
        "start": -785.7371859550476, 
        "status": "DONE"
      }, 
      {
        "id": 1, 
        "lastchange": -778.1088891029358, 
        "stage": "basic_detection", 
        "start": -785.5271852016449, 
        "status": "DONE"
      }, 
      {
        "id": 2, 
        "lastchange": -774.8934471607208, 
        "stage": "identifiaction", 
        "start": -778.1033251285553, 
        "status": "DONE"
      }, 
      {
        "id": 3, 
        "lastchange": -761.449187040329, 
        "stage": "head_orientation_gaze_action_units", 
        "start": -774.8882849216461, 
        "status": "DONE"
      }, 
      {
        "id": 4, 
        "lastchange": -757.6228020191193, 
        "stage": "facial_expressions", 
        "start": -761.4437031745911, 
        "status": "DONE"
      }, 
      {
        "id": 5, 
        "lastchange": -750.1808030605316, 
        "stage": "create_json", 
        "start": -757.6172919273376, 
        "status": "DONE"
      }
    ], 
    "status": "DONE", 
    "uuid": "7a3d50582e8b4781b5eedd7a80307e5b"
  }, 
  "status": "OK"
}
```

Aggregated results
==================
`[GET] /results_agg/{uuid}`
`[GET] /results_agg/{uuid}/start1,end1;start2,end2`
 
 Returns aggregated result for time intervals. If no interval is specified, the results are aggregated accross the whole video.

 Response:
 ```
{
  "@context": "http://senpy.cluster.gsi.dit.upm.es/api/contexts/Results.jsonld", 
  "@type": "results", 
  "analysis": [
    "videoAnalysis_category", 
    "videoAnalysis_vad"
  ], 
  "entries": [
    {
      "@id": "shortTest.mp4#time=0.0,inf", 
      "emotions": [
        {
          "@type": "emotionSet", 
          "onyx:hasEmotion": [
            {
              "@type": "emotion", 
              "onyx:hasEmotionCategory": "big6:happiness", 
              "onyx:hasEmotionIntensity": 0.14045661580052254
            }
          ], 
          "prov:wasGeneratedBy": "videoAnalysis_category"
        }, 
        {
          "@type": "emotionSet", 
          "onyx:hasEmotion": {
            "@type": "emotion", 
            "pad:arousal": -0.27507430185843756, 
            "pad:pleasure": 0.086211333597895456
          }, 
          "prov:wasGeneratedBy": "videoAnalysis_vad"
        }
      ]
    }
  ]
}
 ```

Per-frame information
=====================
`[GET] /results/{uuid}`

Returns analysis results in JSON format. 
The JSON files contain these values for frames and pernsons (not all information has to be present for each `[frame_id][face_id]`):
  * `[frame_id][face_id]['action_units']` -- binary presence and "strength" of facial atcion units 1,2,4,5,6,7,9,10,12,14,15,17,20,23,25,26,45,28 - see https://www.cs.cmu.edu/~face/facs.htm
  * `[frame_id][face_id]['emotions']` -- probabilities of facial expression - the order is anger, disgust, fear, smile, sad, surprised, neutral
  * `[frame_id][face_id]['age']` -- estimated age in years
  * `[frame_id][face_id]['gender']` -- estimated gender [female_prob, male_prob]
  * `[frame_id][face_id]['bounding_box']` -- facial bounding box position in pixels
  * `[frame_id][face_id]['gaze_left']` -- 3D gaze direction vector of left eye; [0,0,-1] is directly into camera
  * `[frame_id][face_id]['gaze_right']` -- 3D gaze direction vector of right eye
  * `[frame_id][face_id]['head_pose']` -- Head pose represented as rotation vector [rot)x, rot_y, rot_z] -- see https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula or http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#void Rodrigues(InputArray src, OutputArray dst, OutputArray jacobian)
  * `[frame_id][face_id]['landmarks']` -- 68 facial landmarks [x,y] -- see https://cdn-images-1.medium.com/max/800/1*AbEg31EgkbXSQehuNJBlWg.png

