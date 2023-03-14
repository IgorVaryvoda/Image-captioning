# Image captioning with BLIP and BLIP2
Powered by replicate

## Getting started
1. Get a [Replicate API key](https://replicate.com/account)
2. Add your API key to .env.example. Rename it to .env (mv .example.env .env)
3. Install dependencies

```pip3 install python-dotenv```

```pip3 install replicate```
4. Run it

```python3 main.py --i /home/images --model blip2```


Options:
- `--i` - image folder location
- `--model` - blip or blip2

Results are saved into the 'results' folder.