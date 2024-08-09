from court_detection_net import CourtDetectorNet
import pandas as pd
from bounce_detector import BounceDetector
from person_detector import PersonDetector
from ball_detector import BallDetector
from utils import scene_detect, read_video, write_video, pipline
import argparse
import torch


if __name__ == '__main__':
    torch.cuda.empty_cache() 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--court_kps_det_model', default='resnet', type=str, help="choose from 'resnet', 'tracknet'")
    parser.add_argument('--person_det_model', default='yolo', type=str, help="choose from 'yolo', 'fasterrcnn'")
    parser.add_argument('--ball_det_model', default='tracknet', type=str, help="choose from 'yolo', 'tracknet'")    
    parser.add_argument('--path_bounce_model', default='./models/ctb_regr_bounce.cbm', type=str, help='path to pretrained model for bounce detection')
    parser.add_argument('--path_input_video', default='./input_videos/input_video1.mp4', type=str, help='path to input video')
    parser.add_argument('--path_output_video_dir', default='./output_videos', type=str, help='path to output video dir')
    args = parser.parse_args()
    
    bounces=None
    ball_track=None
    homography_matrices=None
    kps_court=None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device =',device)
    frames, fps = read_video(args.path_input_video)  # frames[0].shape=(720,  1280, 3)
    scenes = scene_detect(args.path_input_video)    # [[ , ]]

    print('court keypoints detection')
    court_detector = CourtDetectorNet(model_type=args.court_kps_det_model, device=device)
    homography_matrices, kps_court = court_detector.infer_model(frames)

    print('person detection')
    person_detector = PersonDetector(model_type=args.person_det_model, device=device)
    persons_top, persons_bottom = person_detector.track_players(frames, homography_matrices, filter_players=True)
    
    print('ball detection')
    ball_detector = BallDetector(model_type=args.ball_det_model, device=device)
    ball_track = ball_detector.infer_model(frames) # ball_track: list[tuple[None, None]]

    df = pd.DataFrame(ball_track[2:], columns=['x', 'y'])
    # 使用插值函数填补缺失值
    df= df.interpolate().bfill().ffill()
    ball_track[2:] = list(df.itertuples(index=False, name=None))

    
    # bounce detection
    bounce_detector = BounceDetector(args.path_bounce_model)
    x_ball = [x[0] for x in ball_track]
    y_ball = [x[1] for x in ball_track]
    bounces = bounce_detector.predict(x_ball, y_ball)

    imgs_res = pipline(frames, scenes, bounces, ball_track, homography_matrices, kps_court, persons_top, persons_bottom,
                    draw_trace=True)
    
    path_output_video = f"{args.path_output_video_dir}/output_video{args.path_input_video[-5]}_{args.court_kps_det_model}4court_{args.person_det_model}4person_{args.ball_det_model}4ball.avi"

    write_video(imgs_res, fps, path_output_video)





