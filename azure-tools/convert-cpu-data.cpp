/*
 * Converts Azure's CPU data to per user data.
 *
 * Author: Liran Funaro <liran.funaro@gmail.com>
 *
 * Copyright (C) 2006-2018 Liran Funaro
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#include <iostream>
#include <fstream>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <set>

using namespace std;

#define SZ (1<<12)
#define ID_SET_SZ (1<<25)
#define TOTAL_LINES (10000000)


int main(int argc, char *argv[]) {
	if (argc < 3) {
		cout << "Must specify VM two arguments:" << endl
			 << "   1. ID set file path" << endl
			 << "   2. CPU data directory path" << endl;
		exit(1);
	}
	const char* vm_id_set_fname  = argv[1];
	const char* cpu_data_dname  = argv[2];


    char* buff = new char[SZ];
    char* outfile_name = new char[SZ];
    strcpy(outfile_name, cpu_data_dname);
    char* outfile_id = outfile_name + strlen(outfile_name);
    if (*outfile_id != '/') {
    	*(outfile_id++) = '/';
    	*outfile_id = '\0';
    }

    unsigned int line_count = 0;
    unsigned int useful_line_count = 0;
    unsigned int ev_last_line = 0;
    unsigned int evaluate_each = 1<<15;

    time_t start_time, ev_time;
    start_time = time (NULL);
    ev_time = start_time;

    cout.imbue(std::locale(""));

    set<string> vm_ids;
    std::ifstream setfile;
    setfile.open(vm_id_set_fname, std::ios_base::app);
    while (setfile.getline(buff, SZ)) {
        for(unsigned int i=0; i<SZ; i++) {
            if (buff[i] == '/')
                buff[i] = '_';
            else if (buff[i] == '\0')
                break;
        }
        vm_ids.insert(buff);
    }
    setfile.close();

    while (cin.getline(buff, SZ)) {
        line_count++;

        unsigned int i1, i2;
        for(i1=0; i1<SZ; i1++) {
            if (buff[i1] == ',') {
                buff[i1] = '\0';
                break;
            }
        }
        if (i1 == SZ) break;
        i1++;

        for(i2=i1; i2<SZ; i2++) {
            if (buff[i2] == '/')
                buff[i2] = '_';
            else if (buff[i2] == ',') {
                buff[i2] = '\0';
                break;
            }
        }
        if (i2 == SZ) break;
        i2++;

        if (vm_ids.find(buff+i1) == vm_ids.end())
            continue;

        strcpy(outfile_id, buff+i1);

        std::ofstream outfile;
        outfile.open(outfile_name, std::ios_base::app);
        outfile << buff << "," << (buff+i2) << std::endl;

        useful_line_count++;

        if (line_count % evaluate_each == 0) {
            time_t cur_time = time (NULL);
            time_t t = cur_time - ev_time;
            if (t > 0) {
                time_t gt = cur_time - start_time;

                double lps = (double)(line_count - ev_last_line) / (double)t;
                double global_lps = (double)(line_count) / (double)gt;
                double eta = double(TOTAL_LINES - line_count) / global_lps;
                
                cout << "\r  [" << gt << "] "
                     << "[Line: " << std::fixed << line_count << "] "
                     << "[Useful: " << std::fixed << useful_line_count << "] "
                     << "[Throughput: " << std::setprecision(2) << global_lps << " ("
                     << std::setprecision(2) << lps << ") lines/second] "
                     << "[ETA: " << std::setprecision(2) << eta << " seconds]      ";

                ev_time = cur_time;
                ev_last_line = line_count;
            }
        }
    }

    cout << endl;

    delete[] buff;
    delete[] outfile_name;
    return 0;
}