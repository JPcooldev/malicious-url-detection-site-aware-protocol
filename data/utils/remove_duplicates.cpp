// Fast, unlike python version
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <functional>
#include <filesystem>
#include <sstream>

namespace fs = std::filesystem;

// Extract everything up to the first comma
static inline std::string extract_url(const std::string &line) {
    auto pos = line.find(',');
    return (pos == std::string::npos) ? line : line.substr(0, pos);
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <input.csv> <output.csv> <num_buckets> <tmp_dir>\n";
        return 1;
    }
    const std::string input_csv  = argv[1];
    const std::string output_csv = argv[2];
    const int NUM_BUCKETS        = std::stoi(argv[3]);
    const fs::path TMP_DIR       = argv[4];

    // Ensure temp directory exists
    fs::create_directories(TMP_DIR);

    // Phase 1: open bucket files
    std::vector<std::ofstream> buckets;
    buckets.reserve(NUM_BUCKETS);
    for (int i = 0; i < NUM_BUCKETS; ++i) {
        fs::path p = TMP_DIR / ("bucket_" + std::to_string(i) + ".tmp");
        buckets.emplace_back(p);
        if (!buckets.back().is_open()) {
            std::cerr << "Error opening " << p << "\n";
            return 1;
        }
    }

    // Phase 1: distribute lines
    std::ifstream in(input_csv);
    if (!in.is_open()) {
        std::cerr << "Error opening " << input_csv << "\n";
        return 1;
    }
    std::string header;
    std::getline(in, header);

    std::cout << "Phase 1: Bucketing lines...\n";
    std::string line;
    size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        auto url = extract_url(line);
        size_t h = std::hash<std::string>{}(url);
        int idx = int((h & 0x7fffffff) % NUM_BUCKETS);
        buckets[idx] << line << "\n";
        if (line_no % 1'000'000 == 0) {
            std::cout << "\r  Bucketed " << line_no << " lines" << std::flush;
        }
    }
    std::cout << "\r  Bucketed " << line_no << " lines. Done.\n";
    in.close();
    for (auto &ofs : buckets) ofs.close();

    // Phase 2: dedupe each bucket
    std::ofstream out(output_csv);
    if (!out.is_open()) {
        std::cerr << "Error opening " << output_csv << "\n";
        return 1;
    }
    out << header << "\n";

    std::cout << "Phase 2: Deduping buckets...\n";
    for (int i = 0; i < NUM_BUCKETS; ++i) {
        fs::path p = TMP_DIR / ("bucket_" + std::to_string(i) + ".tmp");
        std::ifstream bin(p);
        if (!bin.is_open()) continue;

        std::unordered_set<std::string> seen;
        seen.reserve(1'000'000);

        size_t kept = 0;
        while (std::getline(bin, line)) {
            auto url = extract_url(line);
            if (seen.insert(url).second) {
                out << line << "\n";
                ++kept;
            }
        }
        bin.close();
        fs::remove(p);
        std::cout << "\r  Bucket " << (i+1) << "/" << NUM_BUCKETS
                  << " -> kept " << kept << " uniques" << std::flush;
    }
    std::cout << "\nDone. Output: " << output_csv << "\n";

    // Clean up temp dir if empty
    fs::remove(TMP_DIR);

    return 0;
}
