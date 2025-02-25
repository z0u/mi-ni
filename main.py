from time import sleep
import modal

app = modal.App("example-get-started")


@app.function()
def square(x):
    print(f"This code is running on a remote worker! {x}**2=...")

    # Interestingly, the global import of `sleep` is available here.
    # For packages that are only available on the worker, you can import them
    # in the function body.
    sleep(1)
    return x**2


@app.local_entrypoint()
def main():
    print("the square is", square.remote(4))
