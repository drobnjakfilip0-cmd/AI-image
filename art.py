from diffusers import StableDiffusionPipeline
import torch

def main():

    # 🔥 MODEL KOJI UVEK RADI — runwayml/stable-diffusion-v1-5
    model_id = "runwayml/stable-diffusion-v1-5"

    print("Učitavam model, sacekaj malo...")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None  # isključuje safe checker, brže je i radi bez greške
    )

    # koristi GPU ako postoji
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print("Koristim CUDA (GPU)")
    else:
        print("Nema GPU — ide CPU, malo sporije")

    # tvoj prompt
    prompt = "a majestic fox in a forest, ultra realistic, 4k"

    print("Generišem sliku...")

    image = pipe(prompt).images[0]

    # snimi fajl
    output_path = "output.png"
    image.save(output_path)

    print(f"Slika je sačuvana kao {output_path}")

if __name__ == "__main__":
    main()
