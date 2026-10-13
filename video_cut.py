import os
from moviepy import VideoFileClip
import moviepy.video.fx as vfx


def process_video(
    input_path, output_path, crop_bottom_ratio=0.10, speed_factor=0.5
):
  """裁切影片底部指定比例，並放慢播放速度 (例如 speed_factor=0.5 為半速慢動作)"""
  if not os.path.exists(input_path):
    print(f"❌ 錯誤：找不到檔案 -> {input_path}")
    return

  # 讀取影片
  clip = VideoFileClip(input_path)
  w, h = clip.size

  # 計算高度 (保留上方 90%)
  new_width = w
  new_height = int(h * (1.0 - crop_bottom_ratio))

  # 確保寬高為偶數 (避免 H.264 編碼器報錯)
  if new_width % 2 != 0:
    new_width -= 1
  if new_height % 2 != 0:
    new_height -= 1

  print(f"原解析度: {w}x{h} -> 裁切後解析度: {new_width}x{new_height}")
  print(
      f"原時長: {clip.duration:.2f}s -> 減速後時長:"
      f" {clip.duration / speed_factor:.2f}s (倍速: {speed_factor}x)"
  )

  # MoviePy 2.x 同時套用「裁切」與「慢速播放」特效
  processed_clip = clip.with_effects([
      vfx.Crop(x1=0, y1=0, width=new_width, height=new_height),
      vfx.MultiplySpeed(factor=speed_factor),  # 0.5 代表放慢一倍
  ])

  # 輸出影片
  processed_clip.write_videofile(
      output_path, codec="libx264", audio_codec="aac", threads=4
  )

  # 釋放資源
  clip.close()
  processed_clip.close()
  print(
      f"\n 處理完成！已成功裁掉底部 10% 並放慢至"
      f" {speed_factor}x，新影片儲存至：\n{output_path}"
  )


if __name__ == "__main__":
  input_video = os.path.join("assets", "PixVerse_Cropped.mp4")
  output_video = os.path.join("assets", "PixVerse_Clean_Slow.mp4")

  # 裁切底部 10% 且放慢為 0.5 倍速 (可自由調整 0.5 ~ 0.8)
  process_video(
      input_video, output_video, crop_bottom_ratio=0.20, speed_factor=0.5
  )