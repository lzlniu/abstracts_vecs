#include <fstream>
#include <iostream>
#include <string>
#include <mutex>
#include <vector>
#include <thread>
#include <unordered_set>
#include "boost/regex.hpp"
#include "boost/iostreams/filter/gzip.hpp"
#include "boost/iostreams/filtering_stream.hpp"

std::mutex mtx;
//const unsigned int processor_count = std::thread::hardware_concurrency();

std::unordered_set<int> load_pmids(const char* file_name) {
	std::unordered_set<int> pmids_hash;
	std::ifstream file(file_name);
	std::string line;
	while(std::getline(file, line)) pmids_hash.insert(std::stoi(line));
	file.close();
	return pmids_hash;
}

void show_hash(std::unordered_set<int> hash) {
	for (int i : hash) std::cout << i << std::endl;
}

void gunzip_and_match_abstract(const char* filename, const std::unordered_set<int> &pmids, std::ofstream &ofile) {
	std::ifstream igzfile(filename, std::ios::binary);
	try {
		boost::iostreams::filtering_istream in;
		in.push(boost::iostreams::gzip_decompressor());
		in.push(igzfile);
		std::string line;
		boost::smatch match;
		boost::regex pattern("^PMID:([0-9]+)(.*?)(\\t|$)(.*?)(\\t|$)(.*?)(\\t|$)(.*?)(\\t|$)(.*?)(\\t|$)(.*?)$");
		while (std::getline(in, line)) {
			if (boost::regex_search(line, match, pattern)) {
				int pmid = std::stoi(match[1]);
				if (pmids.find(pmid) != pmids.end()) {
					if (match[12].matched) {
						mtx.lock();
						ofile << pmid << "\t" << match[12] << std::endl;
						mtx.unlock();
					} else {
						mtx.lock();
						ofile << pmid << std::endl;
						mtx.unlock();
					}
				}
			}
		}
	} catch (const boost::iostreams::gzip_error& e) {
		std::cout << e.what() << '\n';
	}
	igzfile.close();
}

int main(int argc, char* argv[]) {
	if (argc < 4) {
		std::cout << "ERROR: You input " << argc-1 << " argument. Need at least 3 argument: [1]PMID list file; [2]Output file; [3]Input pubmedXXXXXXX.tsv.gz file(s)" << std::endl;
		return 1;
	}
	const std::unordered_set<int> pmids_list = load_pmids(argv[1]); // argv[1] is the file which contain all input pmids
	std::ofstream output_file(argv[2]); // argv[2] is the output file name (and path) which specify by the user
	std::vector<std::thread> threads;
	for (int i = 3; i < argc; i++) {
		threads.push_back(std::thread(gunzip_and_match_abstract, argv[i], std::ref(pmids_list), std::ref(output_file)));
		//gunzip_and_match_abstract(argv[i], pmids_list, output_file); // argv[3] and argv[i>3] are the input pubmedXXXX.tsv.gz files
	}
	for (auto &th : threads) th.join();
	output_file.close();
	return 0;
}

