def make_device(device_name):
    if device_name == "keyboard":
        from device.keyboard_interface import KeyboardInterface

        device = KeyboardInterface()

    else:
        raise Exception(
            f"Unrecognized device: {device}. Choose one of 'keyboard'"
        )
    device.print_usage()
    return device
