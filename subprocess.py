from ultralytics import YOLO
import subprocess
import os

ROOT_DIR = os.getcwd()

print(ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR,'data')
TRAINING_DIR = os.path.join(ROOT_DIR,'training')


# Your `init_yaml` function here
def init_yaml(kfold_number=1):
    data = {
        'path': os.path.join(DATA_DIR,str(kfold_number)),
        'train': 'train/images',
        'val': 'val/images',
        'names': {
            0: 'wound'
        }
    }
    # Save data to a temporary YAML file
    yaml_path = os.path.join(ROOT_DIR, f"config.yaml")
    with open(yaml_path, 'w') as yaml_file:
        import yaml
        yaml.dump(data, yaml_file)
    return yaml_path

if __name__ == '__main__':
    for i in range(1, 6):
        yaml_path = init_yaml(i)

        args = dict(
            model="yolov8x-seg.pt",
            data=yaml_path,  # Pass the generated YAML file
            project=TRAINING_DIR,
            name="train_test_" + str(i),
            epochs=1,
            imgsz=512,
            batch=16,
            deterministic=True,
            plots=True,
            seed=1401,
            close_mosaic=0,
            augment=False,
            hsv_h=0,
            hsv_s=0,
            hsv_v=0,
            degrees=0,
            translate=0,
            scale=0,
            shear=0.0,
            perspective=0,
            flipud=0,
            fliplr=0,
            bgr=0,
            mosaic=0,
            mixup=0,
            copy_paste=0,
            erasing=0,
            crop_fraction=0
        )


        command = ["yolo", "train"] + [f"{k}={v}" for k, v in args.items()]
        try:
            # Run the subprocess
            subprocess.run(command, check=True)
        except FileNotFoundError:
            print("YOLO CLI not found. Ensure it is installed and available in your PATH.")
        except subprocess.CalledProcessError as e:
            print(f"Error during YOLO training: {e}")
