import sys

from real_robot.utils.realsense import RSDevice, get_connected_rs_devices

if __name__ == "__main__":
    device_sns = get_connected_rs_devices()

    if len(device_sns) == 0:
        print("No connected RealSense device found!", file=sys.stderr)
        exit(0)

    print(f"Found {len(device_sns)} RealSense device", file=sys.stderr)
    device_sn = device_sns[0]

    if len(device_sns) >= 2:
        print("\nDevice index: device serial number", file=sys.stderr)
        for i, device_sn in enumerate(device_sns):
            print(f"\t{i}: {device_sn=}", file=sys.stderr)
        print(
            "Please enter the index of device to print info: ",
            flush=True,
            file=sys.stderr,
        )
        device_idx = int(input())
        assert 0 <= device_idx <= len(device_sns) - 1, f"Invalid {device_idx=}"
        device_sn = device_sns[device_idx]

    device = RSDevice(device_sn)
    device.print_device_info()

    print(f"Finished printing device info for {device!r}", file=sys.stderr)
