from statistics import mode
import trajectory
import detectron_predict
import numpy as np
import cv2


class AnomalyDetector:
    def __init__(self):
        pass

    @staticmethod
    def get_video(video_path):
        cap = cv2.VideoCapture(video_path)
        count = 0
        tensor = None
        while cap.isOpened():
            count += 1
            print('frame number: ', count)
            ret, frame = cap.read()
            if ret:
                if tensor is None:
                    tensor = frame[:, np.newaxis]
                else:
                    tensor = np.concatenate([tensor, frame], axis=0)

        return tensor

    @staticmethod
    def motion_tracking(video_path, output_pkl_path=None):
        return trajectory.App(video_path, output_pkl_path=output_pkl_path).run()

    @staticmethod
    def box_regression(video_tensor, frame_number):
        _, bbox_dict = detectron_predict.predict(video_tensor[frame_number, :, :, :])
        return bbox_dict

    @staticmethod
    def point_box_intersection(point, box):
        px, py = point
        bx1, by1, bx2, by2 = box
        return bx1 <= px <= bx2 and by1 <= py <= by2

    def main(self, video_path, trajectory_sample=1):
        trajectories = self.motion_tracking(video_path)
        video_tensor = self.get_video(video_path)
        trajectory_classes = {}
        for trajectory_idx, (trajectory, start_frame) in enumerate(trajectories):
            sample_classes = []
            for sampled_idx in range(trajectory_sample):
                sampled_point = trajectory[sampled_idx]
                box_dict = self.box_regression(video_tensor, start_frame)
                for cls, boxes in box_dict.items():
                    for box in boxes:
                        if self.point_box_intersection(sampled_point, box):
                            sample_classes.append(cls)
                            break
                    else:
                        continue  # inner loop didn't break, keep iterating
                    break  # inner loop broke
            sample_majority_class = mode(sample_classes)
            trajectory_classes[trajectory_idx] = sample_majority_class

# if __name__ == '__main__':
print('Started')
ad = AnomalyDetector()
ad.main(video_path='/home/work/allVids/PEDESTRIANS_11_13.25-13.45.avi')