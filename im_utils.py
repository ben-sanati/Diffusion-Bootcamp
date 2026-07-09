import numpy as np
from skimage.data import astronaut
from skimage.transform import resize



def noiser(steps=50, noising_frames=[]):
    image = astronaut()
    image = resize(image, (128,128), anti_aliasing=True)
    for amount_of_noise in np.linspace(0,1,steps):
        noise = np.random.normal(
            0.5,
            0.25,
            image.shape
        )

        noisy_image = (
            (1-amount_of_noise)*image
            +
            amount_of_noise*noise
        )

        noising_frames.append(
            np.clip(
                noisy_image,
                0,
                1
            )
        )
    return noising_frames
