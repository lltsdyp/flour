from huggingface_hub import snapshot_download

repo_id = input("Enter model id:")

snapshot_download(
    repo_id=repo_id,
    local_dir="../models/"+repo_id.split('/')[-1],
)