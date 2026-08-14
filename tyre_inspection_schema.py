"""
JSON Schema untuk structured output inspeksi kelayakan vulkanisir ban.
Format: OpenAI-compatible `response_format: json_schema` (strict mode).

Catatan kompatibilitas:
- $ref/$defs dan nullable union ("type": ["object","null"]) TIDAK didukung
  konsisten di semua provider (termasuk beberapa jalur Gemini via OpenRouter),
  jadi skema ini SENGAJA di-inline dan menghindari null union.
- Strict mode mewajibkan SEMUA property ada di "required" dan
  "additionalProperties": false di setiap object level.
- Kalau is_tyre = false (bukan ban), field "kriteria" tetap WAJIB diisi --
  instruksikan di system prompt agar semua sub-kriteria diisi
  status "tidak_terlihat" dan alasan "gambar bukan ban".
"""

SYSTEM_PROMPT = """Kamu adalah sistem AI inspeksi visual ban kendaraan berat untuk menentukan
kelayakan vulkanisir (retreading). Kamu HARUS memverifikasi sendiri isi gambar -- jangan
mengasumsikan gambar adalah ban kecuali kamu benar-benar melihat objek yang menyerupai ban.

LANGKAH 0 -- VERIFIKASI OBJEK (WAJIB, PALING PERTAMA)
Periksa apakah gambar benar-benar menunjukkan ban kendaraan (utuh atau sebagian: tapak,
sidewall, atau casing yang cukup jelas untuk diperiksa).
- Jika gambar BUKAN ban -> "is_tyre": false, "status": "invalid", isi SEMUA sub-kriteria
  dengan status "tidak_terlihat" dan alasan "gambar bukan ban". STOP di sini.
- Jika gambar adalah ban tapi terlalu buram/gelap/sebagian kecil -> "is_tyre": true,
  "status": "tidak_dapat_ditentukan", jelaskan bagian mana yang tidak cukup terlihat.
- Hanya lanjut ke Langkah 1 jika gambar adalah ban DAN cukup jelas untuk diperiksa.

LANGKAH 1 -- PERIKSA SETIAP KRITERIA (nilai: aman / bermasalah / tidak_terlihat)
1. casing_cord -- retakan/sobekan menembus sampai benang/kawat casing? Retak permukaan
   yang tidak menembus cord masih wajar KECUALI lebar & dalam (tetap tandai bermasalah).
2. bulging_deformasi -- ban menggembung/bulging atau sidewall tidak simetris/peyot?
3. bekas_tusukan -- jika ada, apakah sedikit, kecil, dan tidak saling berdekatan?
4. tambalan -- jika ada, apakah sedikit & di lokasi aman, atau banyak di area kritis?
5. ketebalan_tapak -- tapak tipis/botak itu NORMAL untuk vulkanisir, bukan penolakan.

ATURAN KEPUTUSAN
- casing_cord atau bulging_deformasi = "bermasalah" -> "bad" (VETO, mengesampingkan lainnya).
- bekas_tusukan atau tambalan = "bermasalah" (tanpa veto di atas) -> "bad".
- Tidak ada masalah, tapi tapak tipis/botak -> "warning".
  (PENTING: Status 'warning' berarti ban sudah tipis tapi strukturnya aman, sehingga INILAH yang disebut "Kondisi Layak Vulkanisir". Berikan label dan rekomendasi bahwa ban ini layak dilanjutkan ke proses vulkanisir).
- Semua aman termasuk tapak tebal -> "good".
  (PENTING: Status 'good' berarti ban masih bagus dan tebal. JANGAN tulis "layak vulkanisir" pada keterangan/rekomendasinya, melainkan tulis bahwa ban dalam kondisi bagus dan belum perlu divulkanisir).
- Kriteria veto berstatus "tidak_terlihat" -> "tidak_dapat_ditentukan", JANGAN menebak.

LARANGAN
- Jangan mengklaim tahu riwayat vulkanisir sebelumnya atau usia ban.
- Jangan beri confidence_percent tinggi (>80) untuk kriteria yang tidak_terlihat/meragukan.
- Jawab HANYA JSON sesuai skema yang diberikan, tanpa teks lain di luar JSON, tanpa
  menampilkan proses berpikir (reasoning/think) di dalam field manapun.
"""


def _kriteria_item():
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {
                "type": "string",
                "enum": ["aman", "bermasalah", "tidak_terlihat"]
            },
            "alasan": {
                "type": "string",
                "description": "Alasan singkat penilaian kriteria ini."
            }
        },
        "required": ["status", "alasan"]
    }


TYRE_INSPECTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "tyre_inspection_result",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "is_tyre": {
                    "type": "boolean",
                    "description": "True jika gambar benar-benar menunjukkan ban kendaraan."
                },
                "status": {
                    "type": "string",
                    "enum": ["good", "warning", "bad", "tidak_dapat_ditentukan", "invalid"]
                },
                "label": {
                    "type": "string",
                    "description": "Nama kondisi ban dalam Bahasa Indonesia."
                },
                "kriteria": {
                    "type": "object",
                    "additionalProperties": False,
                    "description": (
                        "Wajib diisi selalu. Jika is_tyre=false, isi semua sub-kriteria "
                        "dengan status 'tidak_terlihat' dan alasan 'gambar bukan ban'."
                    ),
                    "properties": {
                        "casing_cord": _kriteria_item(),
                        "bulging_deformasi": _kriteria_item(),
                        "bekas_tusukan": _kriteria_item(),
                        "tambalan": _kriteria_item(),
                        "ketebalan_tapak": _kriteria_item(),
                    },
                    "required": [
                        "casing_cord",
                        "bulging_deformasi",
                        "bekas_tusukan",
                        "tambalan",
                        "ketebalan_tapak",
                    ]
                },
                "confidence_percent": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100
                },
                "message": {
                    "type": "string",
                    "description": "Penjelasan singkat kondisi ban (1-2 kalimat)."
                },
                "recommendation": {
                    "type": "string",
                    "description": "Saran tindak lanjut yang spesifik."
                }
            },
            "required": [
                "is_tyre",
                "status",
                "label",
                "kriteria",
                "confidence_percent",
                "message",
                "recommendation"
            ]
        }
    }
}
