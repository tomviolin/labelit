import cv2
import numpy as np

def put_text_rect(img, text, position, font_face, font_size, rect_color, font_color, font_thickness, tightness=0, anchor='baseline-left'):
    (ptx, pty) = position
    # Get the text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font_face, font_size, font_thickness)
    
    # Adjust height for baseline
    text_height += baseline
    
    # Calculate rectangle coordinates
    top_left = (ptx, pty - text_height + tightness)
    bottom_right = (ptx + text_width, pty + baseline - tightness)

    # adjust position based on anchor. every anchor is based on baseline-left
    # baseline-left is the default anchor; all other anchors are relative to this one
    pos_adjust = (0, 0)
    if anchor == 'baseline-left':
        pass
    elif anchor == 'baseline-center':
        pos_adjust = (-text_width//2, 0)
    elif anchor == 'baseline-right':
        pos_adjust = (-text_width, 0)
    elif anchor == 'top-left':
        pos_adjust = (0, text_height)
    elif anchor == 'top-center':
        pos_adjust = (-text_width//2, text_height)
    elif anchor == 'top-right':
        pos_adjust = (-text_width, text_height)
    elif anchor == 'center-left':
        pos_adjust = (0, (text_height-baseline)//2)
    elif anchor == 'center':
        pos_adjust = (-text_width//2, (text_height-baseline)//2)
    elif anchor == 'center-right':
        pos_adjust = (-text_width, (text_height-baseline)//2)
    elif anchor == 'bottom-left':
        pos_adjust = (0, -baseline)
    elif anchor == 'bottom-center':
        pos_adjust = (-text_width//2, -baseline)
    elif anchor == 'bottom-right':
        pos_adjust = (-text_width, -baseline)

    # Adjust the position
    top_left = (top_left[0] + pos_adjust[0], top_left[1] + pos_adjust[1])
    bottom_right = (bottom_right[0] + pos_adjust[0], bottom_right[1] + pos_adjust[1])

    # Draw the rectangle
    cv2.rectangle(img, top_left, bottom_right, rect_color, cv2.FILLED)
    
    # Put the text on the image
    cv2.putText(img, text, (ptx+pos_adjust[0], pty+pos_adjust[1]), font_face, font_size, font_color, font_thickness, cv2.LINE_AA)

    return img

# Example usage
if __name__ == "__main__":
    img = np.zeros((600, 500, 3), np.uint8)  # Black image
    for i in range(50, 500, 40):
        cv2.line(img, (0, i), (500, i), (255, 255, 255), 1)
    cv2.line(img, (250, 0), (250, 500), (255, 255, 255), 1) 
    for posy in range(50, 500, 40):
        position = (250, posy)
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.5# +posy/500 # Font size increases with y
        rect_color = (0, 255, 128)  # Green rectangle
        font_color = (0, 0, 0)  # White text
        font_thickness = 1
        pord = (posy - 50) // 40
        anchor = ['baseline-left', 'baseline-center', 'baseline-right', 'top-left', 'top-center', 'top-right', 'center-left', 'center', 'center-right', 'bottom-left', 'bottom-center', 'bottom-right'][pord % 12]

        img_with_text = put_text_rect(img, f"110101 Hellogy, World! {font_size} {anchor}", position, font_face, font_size, rect_color, font_color, font_thickness,0, anchor)
        cv2.drawMarker(img_with_text, position, (0,0,255), cv2.MARKER_CROSS, 10, 1)

    cv2.imshow('Image with Text', img_with_text)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

