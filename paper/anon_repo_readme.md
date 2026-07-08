<div align="center">
<h1>Darwinian Memory: A Training-Free Self-Regulating Memory 

System for GUI Agent Evolution</h1>
</div>

<br>
<div align="center">
<img src='img/head.png' width='65%'>
</div>

<div align="center">
Performance Overview of DMS. In multi-app GUI scenarios, DMS consistently boosts Accuracy and Stability across diverse general-purpose models while reducing task latency.
</div>

## 💡 Introduction
Welcome to **DMS**, in this work,we propose the Darwinian Memory System (DMS), a self-evolving architecture that constructs memory as a dynamic ecosystem governed by the law of **survival of the fittest**.
<div align="center">
<img src='img/main.png' width='65%'>
</div>

## 🎯 Current Results
To validate the effectiveness and robustness of our method, we conducted extensive experiments on authoritative and challenging benchmark:

* **[AndroidWorld](https://github.com/google-research/android_world)**: A rigorous environment for autonomous Android agents. We tested the agent's ability to execute complex, multi-step tasks in a dynamic real-world Android environment.

### Main Results
<div align="center">
<img src='img/mainresult.png' width='65%'>
</div>

### Success Rate Detail across rounds and difficulties
<div align="center">
<img src='img/detailresults.png' width='65%'>
</div>

## 🛠️ Installation & Setup

### Step 1: Install ADB
First, install **[Android Debug Bridge (ADB)](https://developer.android.com/tools/adb)** to allow the code to control the Android device.
* Follow the official documentation to complete the installation.
* **Important:** Ensure you **enable Developer Mode** and turn on **USB Debugging** on your phone or emulator.

### Step 2: Setup AndroidWorld Environment
Follow the instructions in the **[AndroidWorld README](https://github.com/google-research/android_world)** to correctly set up the environment.

To verify that your environment is set up correctly, you can use the following command to check the emulator status:
```bash
~/Library/Android/sdk/emulator/emulator -avd $EMULATOR_NAME -no-snapshot -grpc 8554
```

### Step 3: Add the DMS Adapter

Create a new file named agent_adapter.py (or similar) in the directory android_world/agents/ and add the adapter code.

### Step 5: Start Evaluation

Finally, you can start the test using the command below:

```Bash
python run.py --agent_name=xxx
```
Note: For the first run, you might need to use the --perform_emulator_setup parameter to help initialize apps on the emulator, though this is not always mandatory.

We are continuously optimizing and updating the codebase.


