import time
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model as module_arch
from parse_config import ConfigParser
from torchvision import transforms


to_image = transforms.Compose([transforms.ToPILImage()])


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        batch_size=1,
        fine_size=0,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    test_output_dir = config.log_dir / "test_output/"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_dir in data_loader.dataset.dataset_dir_list:
        full_dir = test_output_dir / dataset_dir
        full_dir.mkdir(parents=True, exist_ok=True)
        low_dir = full_dir / str(data_loader.dataset.low_dir)[1:-1]
        low_dir.mkdir(parents=True, exist_ok=True)

    total_time = 0

    for i, (input_img, name) in enumerate(tqdm(data_loader)):
        name = name[0]
        torch.cuda.empty_cache()
        input_img = input_img.to(device)

        input_img = input_img[:, :, :(input_img.shape[2]//2)*2, :(input_img.shape[3]//2)*2]

        torch.cuda.synchronize()
        # start.record()
        start = time.perf_counter()
        generated_img = model(x=input_img, test_scale_shift=True)
        # end.record()
        torch.cuda.synchronize()
        total_time += time.perf_counter() - start
        # total_time += start.elapsed_time(end)

        generated_img = torch.clamp(generated_img, 0, 1)
        generated_img = to_image(torch.squeeze(generated_img.float().detach().cpu()))
        generated_img.save(test_output_dir / name)
        print("write " + name + " successful!")

    print("total time: " + str(total_time))

if __name__ == '__main__':
    with torch.no_grad():
        args = argparse.ArgumentParser(description='PyTorch Template')
        args.add_argument('-c', '--config', default=None, type=str,
                          help='config file path (default: None)')
        args.add_argument('-r', '--resume', default=None, type=str,
                          help='path to latest checkpoint (default: None)')
        args.add_argument('-d', '--device', default=None, type=str,
                          help='indices of GPUs to enable (default: all)')

        config = ConfigParser.from_args(args, training=False)
        main(config)
