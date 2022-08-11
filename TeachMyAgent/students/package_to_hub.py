from huggingface_hub import HfApi, upload_folder
from huggingface_hub.repocard import metadata_eval_result, metadata_save
import tempfile
from pathlib import Path
import subprocess
import json

def package_to_hub(repo_id,
                   ta_config,
                   model_path,
                   mean_reward,
                   std_reward,
                   hyperparameters,
                   token=None
                   ):
    # # Step 1: Clone or create the repo
    repo_url = HfApi().create_repo(
        repo_id=repo_id,
        repo_type="model",
        token=token,
        private=False,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)

        # Step 2: Save the tfjs model
        subprocess.run('tensorflowjs_converter --input_format=tf_saved_model --output_node_names="parkour_walker" ' +
                       f'--saved_model_tags=serve --skip_op_check {model_path}/tf1_save {tmpdirname}',
                       shell=True, check=True)

        # Step 3: Write TeachMyAgent's config fle
        with open(tmpdirname / "ta-config.json", "w") as outfile:
            json.dump(ta_config, outfile)

        # Step 5: Generate the model card
        generated_model_card, metadata = _generate_model_card(ta_config["name"], mean_reward, std_reward, hyperparameters)
        _save_model_card(tmpdirname, generated_model_card, metadata)

        repo_url = upload_folder(
            repo_id=repo_id,
            folder_path=tmpdirname,
            path_in_repo="",
            commit_message=f"Uploading {repo_id}",
            token=token,
        )

    return repo_url

def _generate_model_card(model_name, mean_reward, std_reward, hyperparameters):
    """
    Generate the model card for the Hub
    :param model_name: name of the model
    :mean_reward: mean reward of the agent
    :std_reward: standard deviation of the mean reward of the agent
    :hyperparameters: training arguments
    """
    # Step 1: Select the tags
    metadata = generate_metadata(model_name, mean_reward, std_reward)

    # Transform the hyperparams namespace to string
    # converted_dict = vars(hyperparameters)
    converted_str = str(hyperparameters)
    converted_str = converted_str.split(", ")
    converted_str = '\n'.join(converted_str)

    # Step 2: Generate the model card
    model_card = f"""
  # Deep RL Agent Playing TeachMyAgent's parkour.
  You can find more info about TeachMyAgent [here](https://developmentalsystems.org/TeachMyAgent/).
  
  Results of our benchmark can be found in our [paper](https://arxiv.org/pdf/2103.09815.pdf).
  
  You can test this policy [here](https://huggingface.co/spaces/flowers-team/Interactive_DeepRL_Demo)
  
  ## Results
  Percentage of mastered tasks (i.e. reward >= 230) after 20 millions steps on the Parkour track. 
  
  Results shown are averages over 16 seeds along with the standard deviation for each morphology as well as the aggregation of the 48 seeds in the *Overall* column. 
  
  We highlight the best results in bold.
  
  | Algorithm     | BipedalWalker  | Fish          | Climber      | Overall       |
  |---------------|----------------|---------------|--------------|---------------|
  | Random        | 27.25 (± 10.7) | 23.6 (± 21.3) | 0.0 (± 0.0)  | 16.9 (± 18.3) |
  | ADR           | 14.7 (± 19.4)  | 5.3 (± 20.6)  | 0.0 (± 0.0)  | 6.7 (± 17.4)  |
  | ALP-GMM       | **42.7** (± 11.2)  | 36.1 (± 28.5) | 0.4 (± 1.2)  | **26.4** (± 25.7) |
  | Covar-GMM     | 35.7 (± 15.9)  | 29.9 (± 27.9) | 0.5 (± 1.9)  | 22.1 (± 24.2) |
  | GoalGAN       | 25.4 (± 24.7)  | 34.7 ± 37.0)  | 0.8 (± 2.7)  | 20.3 (± 29.5) |
  | RIAC          | 31.2 (± 8.2)   | **37.4** (± 25.4) | 0.4  (± 1.4) | 23.0 (± 22.4) |
  | SPDL          | 30.6 (± 22.8)  | 9.0 (± 24.2)  | **1.0** (± 3.4)  | 13.5 (± 23.0) |
  | Setter-Solver | 28.75 (± 20.7) | 5.1 (± 7.6)   | 0.0 (± 0.0)  | 11.3 (± 17.9) |

  # Hyperparameters
  ```python
  {converted_str}
  ```
  """
    return model_card, metadata


def generate_metadata(model_name, mean_reward, std_reward):
    """
    Define the tags for the model card
    :param model_name: name of the model
    :mean_reward: mean reward of the agent
    :std_reward: standard deviation of the mean reward of the agent
    """
    metadata = {}
    metadata["tags"] = [
        "sac",
        "deep-reinforcement-learning",
        "reinforcement-learning",
        "teach-my-agent-parkour"
    ]

    # Add metrics
    eval = metadata_eval_result(
        model_pretty_name=model_name,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
        dataset_pretty_name="teach-my-agent-parkour",
        dataset_id="teach-my-agent-parkour"
    )

    # Merges both dictionaries
    metadata = {**metadata, **eval}

    return metadata


def _save_model_card(local_path, generated_model_card, metadata):
    """Saves a model card for the repository.
    :param local_path: repository directory
    :param generated_model_card: model card generated by _generate_model_card()
    :param metadata: metadata
    """
    readme_path = local_path / "README.md"
    readme = ""
    if readme_path.exists():
        with readme_path.open("r", encoding="utf8") as f:
            readme = f.read()
    else:
        readme = generated_model_card

    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme)

    # Save our metrics to Readme metadata
    metadata_save(readme_path, metadata)