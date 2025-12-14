if __name__ == '__main__':
    from ultralytics import YOLO

    #model = YOLO('yolov8s.pt')  # pretrained model
    model = YOLO('best.pt')  # custom trained model
    """
    train_results = model.train(
        data="data.yaml",  # path to dataset YAML
        epochs=4,  # number of training epochs
        imgsz=640,  # training image size
        device="cuda"  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    )
    """
    metrics = model.val(
        data="data.yaml",
        device="cuda",
        verbose=True,
        plots=True,
        visualize=True
    )

    cm = metrics.confusion_matrix

    tp, fp, fn = cm.tp_fp()

    print("TP: ", tp[0])
    print("FP: ", fp[0])
    print("FN: ", fn[0])

    print("Precision: ", tp[0]/(tp[0]+fp[0]))
    print("Recall: ", tp[0]/(tp[0]+fn[0]))
    
    #results = model.track(source='C:\\path\\to\\video\\vol.mov', persist=True, save=True, tracker="botsort.yaml", device='cuda')