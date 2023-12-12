import numpy as np
import matplotlib.pyplot as plt
import os


def visualize_images(npz_file, ax):
    # Function to load and visualize images from an npz file
    data = np.load(npz_file)
    images = data['my_array']
    
    # Plot each array in a column
    for i in range(4):
        ax[i].imshow(images[i].transpose(1, 2, 0))  # Transpose to (32, 32, 3) for RGB
        ax[i].axis('off')

def main():
    path1 = '../SC/SC_2d_full_bg_Fold_0/visuals_step_181000_test_invert/'
    path2 = '../SC/SC_2d_loss_weight_3_Fold_0/visuals_step_199000_test/'
    path3 = '../SC/SC_2d_full_bg_Fold_0/visuals_step_181000_test_steps/'
    # Create subplots
    # fig, axes = plt.subplots(4, 10, figsize=(45, 20))

    # # Visualize each group of images
    # for col, index in enumerate([1,3,6,7,11,15,16,17,18,19]):
    #     npz_file = os.path.join(path1, f'{index}.npz')
    #     visualize_images(npz_file, axes[:, col])

    # plt.tight_layout()
    # # Save the figure as a PNG file
    # plt.savefig('./sc_visualization.png')

    # # Show the plot (optional)
    # plt.show()



    fig, axes = plt.subplots(1, 11, figsize=(45,10))
    print(len(axes))
    for i,index in enumerate([0,100,200,300,400,500,600,700,800,900,999]):

        interm_file = os.path.join(path3, f'{index}.npz')
        data = np.load(interm_file)
        interm = data['my_array']
        print(interm.shape)
        axes[i].imshow(interm.transpose(1, 2, 0))  # Transpose to (32, 32, 3) for RGB
        axes[i].axis('off')
    plt.savefig('./sc_visualization_interm.png')
    plt.show()


if __name__ == '__main__':
    main()
