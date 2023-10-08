image1 = imread('gpu_output_image.jpg');
image2 = imread('output_image.jpg');
result=ssim(image1,image2);
fprintf("SSIM:%.4f ",result);

