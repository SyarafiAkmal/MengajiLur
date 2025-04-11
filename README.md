# Tugas Besar IF3270 Pembelajaran Mesin
> Implementasi Feed Forward Neural Network

## Daftar Isi

- [Deskripsi Umum](#deskripsi-umum)
- [Setup dan Instalasi](#setup-dan-instalasi)
- [Cara Menjalankan Program](#cara-menjalankan-program)
- [Kontributor dan Pembagian Tugasnya](#kontributor-serta-pembagian-tugasnya)

## Deskripsi Umum

Neural Network merupakan salah satu model dalam Machine Learning yang telah banyak digunakan dalam berbagai bidang. Salah satu jenis neural network yang paling mendasar adalah Feed Forward Neural Network (FFNN), yang memiliki arsitektur sederhana di mana informasi mengalir dalam satu arah, dari input layer ke output layer, tanpa loop atau umpan balik.

Dalam tugas besar ini, kami mengimplementasikan FFNN dari awal tanpa menggunakan library seperti TensorFlow atau PyTorch. Implementasi ini bertujuan untuk memberikan pemahaman yang lebih dalam mengenai cara kerja internal neural network.

## Dependencies

- numpy
- matplotlib
- sklearn

Instalasi dengan menggunakan `pip`
```bash
pip install numpy matplotlib sklearn
```

## Setup dan Instalasi

Clone Repository ini dengan menjalankan perintah berikut pada terminal Anda

```bash
  git clone https://github.com/SyarafiAkmal/MengajiLur.git
```

## Cara Menjalankan Program

- Buka file Jupyter Notebook pada directory src/test.ipynb
- Jalankan dengan menekan tombol `Run All`
- Hasil akan tampak di bawah sel kode
- Apabila hendak melakukan pengujian dengan parameter yang berbeda, ubah nilai pada baris kode dengan komentar `Modify for testing`

## Kontributor serta Pembagian Tugasnya

| NIM      | Nama                    | Tugas                                           |
|----------|-------------------------|-------------------------------------------------|
| 13522018 | Ibrahim Ihsan Rasyid    | Regularisasi, notebook test keseluruhan, laporan|
| 13522028 | Panji Sri Kuncara Wisma |Visualisasi, save & load, bonus fungsi aktivasi, fix backpropagation, testing mnist dataset, debug dan fix kode ann, laporan |
| 13522076 | Muhammad Syarafi Akmal  | Kode base ann (forward propagation and backward propagation), bonus fungsi aktivasi, inisialisasi bobot, notebook test partial, laporan |