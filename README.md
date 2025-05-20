# Counterfactual Explanations for Continuous Action Reinforcement Learning  
Official code for the IJCAI 2025 paper:  
["Counterfactual Explanations for Continuous Action Reinforcement Learning"](http://arxiv.org/abs/2505.12701)
by Shuyang Dong, Shangtong Zhang, and Lu Feng

---

## 📖 Citation

If you use this code or build upon it, please cite:

```bibtex
@inproceedings{dong2025counterfactual,
  title={Counterfactual Explanations for Continuous Action Reinforcement Learning},
  author={Shuyang Dong and Shangtong Zhang and Lu Feng},
  booktitle={Proceedings of the 34th International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2025}
}
```

---

## 🚀 Overview

This repository contains code and data for reproducing the experiments in our IJCAI 2025 paper on generating counterfactual explanations in continuous action reinforcement learning (RL).  
We provide complete pipelines for two domains:

- 🩺 **Diabetes Case Study**
- 🚀 **Lunar Lander Case Study**

Each study includes baseline training (PPO), counterfactual generation (TD3), and postprocessing for result analysis.

---

## 🛠️ Setup Instructions

### 🔹 Environment (Python 3.8)

Use `conda` to create isolated environments for each part of the project.

#### Diabetes PPO Training:
```bash
conda create -n CF_diabetic_train_ppo python=3.8
conda activate CF_diabetic_train_ppo
pip install stable-baselines3==1.7.0
pip install gym==0.21.0
pip install torch==2.0.0
pip install pandas==1.5.3
pip install tensorboard
```

#### Lunar Lander PPO Training:
```bash
conda create -n CF_LunarLander_train_ppo_generalize python=3.8
conda activate CF_LunarLander_train_ppo_generalize
pip install stable-baselines3==1.7.0
pip install gym==0.21.0
pip install torch==2.0.0
pip install pandas==1.5.3
pip install tensorboard
```

📦 After installing, replace default packages in your environment’s `site-packages` using the provided `package/` folders.

---

## 📦 Project Structure

```
counterfactual-rl/
├── diabetic_case_study/
│   ├── train_ppo/
│   ├── train_td3/
│   ├── data_postprocess/
│   └── sample_data/
├── lunar_lander_case_study/
│   ├── train_ppo/
│   ├── train_td3/
│   ├── data_postprocess/
│   └── sample_data/
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 💡 How to Reproduce Results

### 🔸 Diabetes Case Study

1. **Train PPO Baseline**
   ```bash
   python diabetic_case_train_ppo.py -arg_patient_type adult -arg_patient_id 7 -arg_cuda 0 -arg_train_step 100000 -arg_callback_step 100000
   ```

2. **Generate Counterfactuals**
   ```bash
   sbatch run_diabetic_exp1.sh  # For single environment
   ```

3. **Postprocess Results**
   Edit and run:
   ```bash
   data_postprocess.py  # Set case_name = 'diabetic'
   ```

---

### 🔸 Lunar Lander Case Study

1. **Train PPO Baseline**
   ```bash
   python openai_case_train_ppo_generalize.py -arg_exp_id 1 -arg_cuda 0 -arg_train_step_each_env 500 -arg_callback_step 500 -arg_train_round 3 -arg_lr 0.0001
   ```

2. **Generate Counterfactuals**
   ```bash
   sbatch run_LL_exp1_double_test.sh
   ```

3. **Postprocess Results**
   Edit and run:
   ```bash
   data_postprocess.py  # Set case_name = 'lunar_lander'
   ```

---

## 📈 Results and Figures

Final analysis results and figures are located in:

- `data_postprocess/Diabetes_case_study/across_rp/final_data/`
- `data_postprocess/lunar_lander/across_rp/final_data/`

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
