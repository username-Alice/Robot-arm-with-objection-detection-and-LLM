# Demo-ArmGPT
Description:
a GUI to input object's conditions, displaying objects available
using YOLOv8-seg model to detect the object.
use vicuna-13B to analyse user's input and output corresponding object
make robotic arm to pick up specific object upon request

Download the dependency (setup)
Installation procedure
1. install pyqt5
2. pyvistaqt
3. ultralytics
4. zivid==2.9.0.2.9.0
5. pip uninstall opencv-python,pip install opencv-python-headless
6. pip install  git+https://github.com/Maximilian-Winter/guidance.git
7. pip install git+https://github.com/Maximilian-Winter/llama-cpp-python.git
8. find /guidance/llms/_llama_cpp.py inside guidance library and replace it with the _llama_cpp.py located at the source folder.
9. install transformers
10. install gpt4-x-vicuna-13B.ggmlv3.q5_0.bin to ./source/model folder (can be found in https://huggingface.co/TheBloke/gpt4-x-vicuna-13B-GGML/tree/main)
P.S. requirements.txt is only for reference, directly installing requirements.txt may cuase error.
Use the program:
1. run python gui.py in terminal

limitation:
- the llm model for prompt analysis is slow
- objects that can be detected are a bit limited and accuracy is not high
- llm accuracy can be improved

P.S. 
the eye-to-hand calibration program is modified. No rtde library or urp script are needed.
use the program:
step 0: install requirements.txt
1 run  ./Demo-ArmGPT/zivid-python-samples/source/applications/advanced/hand_eye_calibration/hand_eye_calibration.py 
2 manually move the robot arm(with checkboard) to a position
3 type any key except 'q'
4 view the image and press 'y' if you would like to save the point cloud and pose of robot arm
5 repeat the above process for 20 times with different positions
6 press 'q' and the program will output a yaml file at dataset folder