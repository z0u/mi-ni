from pathlib import Path


data_dir = Path('/data')


# def create_app(name: str) -> tuple[modal.App, modal.Volume]:
#     image = (
#         modal.Image.debian_slim()
#         .pip_install(*freeze(all=True, local=False))
#         .add_local_python_source('experiment', 'utils')
#     )  # fmt: skip
#     volume = modal.Volume.from_name(name, create_if_missing=True)
#     return modal.App(name, image=image, volumes={data_dir: volume}), volume
