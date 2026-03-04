# PaddleOCR GPU版 (Windows) の構築手順

本プロジェクトでは、デフォルトでCPU版の**PaddleOCR**を使用しています。
GPU版の導入には環境構築のハードルが高く、特定のCUDAバージョンとcuDNN, zlibwapi.dllなどの手動配置が必要となるため、**上級者向け**としてご案内しています。

もし推論速度をさらに向上させたい等の理由でGPU版を利用されたい場合は、以下の手順に従って環境構築を行ってください。

## 1. 既存環境のアンインストール
現在インストールされているCPU版PaddlePaddleをアンインストールします。

```bash
pip uninstall -y paddlepaddle paddlepaddle-gpu
```

## 2. PaddlePaddle GPU版のインストール
お使いのCUDAバージョン（例：CUDA 11.8 や CUDA 12.0 等）に適合した `paddlepaddle-gpu` をインストールします。
> 詳細は [PaddlePaddle 公式サイト](https://www.paddlepaddle.org.cn/install/quick) をご参照ください。

例 (CUDA 11.8 の場合):
```bash
python -m pip install paddlepaddle-gpu==2.6.2.post118 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```

## 3. 必要なDLLファイルの手動配置
Windows環境でPaddlePaddle GPU版を動かす際、`[operator < fill_constant > error]` などの初期化エラー（PreconditionNotMetError）が発生することがよくあります。これは `cudnn64_8.dll` や `zlibwapi.dll` が正しく読み込めていないことに起因します。

これを解消するためには、以下のDLLを手動でダウンロードし、CUDAのインストールディレクトリ（例：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\bin`）へコピーしてください。

### ① cuDNN の手動配置
1. [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive) にアクセスし、NVIDIAアカウントでログインします。
2. インストールしたPaddlePaddleとCUDAのバージョンに適合する cuDNN (例: v8.9.x) をダウンロードし、解凍します。
3. 展開したフォルダ内の `bin` フォルダに含まれる**すべてのDLLファイル**（`cudnn64_8.dll` 等）を、システムにインストールされている CUDA の `bin` ディレクトリ（例: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`）へコピーします。

### ② zlibwapi.dll の手動配置
1. [zLib DLL x64](http://www.winimage.com/zLibDll/zlib123dllx64.zip) をダウンロードして展開します。
2. 展開したフォルダ内にある `dll_x64\zlibwapi.dll` を、先ほどと同じく CUDA の `bin` フォルダ、または `C:\Windows\System32` ヘコピーします。

## 4. 動作確認
以上の配置が完了したら、新しいターミナルを開きなおして、以下のPythonコマンドを実行して動作確認を行います。

```bash
python -c "import paddle; paddle.utils.run_check()"
```

`PaddlePaddle is installed successfully!` のような完了メッセージが表示されれば、環境構築は成功です。以降、システム全体で PaddleOCR の GPU 推論が有効化されます。
