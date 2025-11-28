from .cli import main

if __name__ in {"__main__", "__mp_main__"}:
    main(standalone_mode=True)
