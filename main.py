from statistics import mode
import trajectory
import detectron_predict

class AnomalyDetector:
    def __init__(self):
        pass

    @staticmethod
    def motion_tracking(video_path):
        raise NotImplementedError  # TODO: Support Batel's code

    @staticmethod
    def box_regression(video_path, frame_number):
        # returns {'person' : tensor.tensor((_, 4)), 'vehicle' : tensor.Tensor((_,4))}
        raise NotImplementedError  # TODO: Support Yoad's code

    @staticmethod
    def get_video(video_path):
        raise NotImplementedError  # TODO: Get videos from Object

    @staticmethod
    def point_box_intersection(point, box):
        px, py = point
        bx1, by1, bx2, by2 = box
        return bx1 <= px <= bx2 and by1 <= py <= by2

    def main(self, video_path, trajectory_sample=1):
        trajectories = self.motion_tracking(video_path)
        trajectory_classes = {}
        for trajectory_idx, (trajectory, start_frame) in enumerate(trajectories):
            sample_classes = []
            for sampled_idx in range(trajectory_sample):
                sampled_point = trajectory[sampled_idx]
                box_dict = self.box_regression(start_frame)
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




