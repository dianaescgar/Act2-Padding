import numpy as np
import cv2
import matplotlib.pyplot as plt

# Función que realiza la convolución en una imagen dada un kernel
def convolution(image, kernel, average=False, verbose=False):
    # Verifica si la imagen tiene 3 canales (si es a color)
    if len(image.shape) == 3:
        print("Found 3 Channels : {}".format(image.shape))  # Imprime la forma original de la imagen
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convierte la imagen a escala de grises
        print("Converted to Gray Channel. Size : {}".format(image.shape))  # Muestra el nuevo tamaño de la imagen
    else:
        print("Image Shape : {}".format(image.shape))  # Imprime el tamaño de la imagen si ya es en escala de grises

    print("Kernel Shape : {}".format(kernel.shape))  # Muestra la forma del kernel utilizado

    # Si la opción verbose está activada, muestra la imagen original
    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Image")
        plt.show()

    # Obtiene las dimensiones de la imagen y del kernel
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    # Inicializa la imagen de salida con ceros
    output = np.zeros(image.shape)

    # Calcula el tamaño del padding necesario
    pad_height = int((kernel_row - 1) / 2)  # Altura del padding
    pad_width = int((kernel_col - 1) / 2)  # Ancho del padding

    # Crea una imagen más grande con ceros alrededor para el padding
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    # Inserta la imagen original en la parte central de la imagen con padding
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    # Si verbose está activado, muestra la imagen con padding
    if verbose:
        plt.imshow(padded_image, cmap='gray')
        plt.title("Padded Image")
        plt.show()

    # Aplica la convolución recorriendo la imagen original con padding
    for row in range(image_row):
        for col in range(image_col):
            # Aplica la convolucion en un fragmento de la imagen con padding del mismo tamaño que el kernel
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1] # Divide el valor resultante por el número de elementos en el kernel

    print("Output Image size : {}".format(output.shape))  # Muestra el tamaño de la imagen de salida

    # Si verbose está activado, muestra la imagen resultante después de la convolución
    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
        plt.show()

    return output  # Regresa la imagen resultante de la convolución
