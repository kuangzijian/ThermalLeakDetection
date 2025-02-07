from PIL import Image, ImageSequence

def stitch_gifs_horizontally(gif1_path, gif2_path, output_path):
    gif1 = Image.open(gif1_path)
    gif2 = Image.open(gif2_path)

    frames = []

    for frame_gif1, frame_gif2 in zip(ImageSequence.Iterator(gif1), ImageSequence.Iterator(gif2)):
        new_height = frame_gif1.height
        frame_gif2_resized = frame_gif2.resize((int(frame_gif2.width * new_height / frame_gif2.height), new_height), Image.ANTIALIAS)
        
        new_frame = Image.new('RGBA', (frame_gif1.width + frame_gif2_resized.width, new_height))
        
        new_frame.paste(frame_gif1, (0, 0))
        new_frame.paste(frame_gif2_resized, (frame_gif1.width, 0))
        
        frames.append(new_frame)

    frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0, duration=gif1.info['duration'], disposal=2)

stitch_gifs_horizontally('transition_3_pos.gif', 'transition_3_pos_diff.gif', 'transition_3_pos_compare_stage.gif')
