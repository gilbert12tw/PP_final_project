#include <iostream>
#include <fstream>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <binary_file>" << endl;
        return 1;
    }

    ifstream file(argv[1], ios::binary);
    if (!file) {
        cerr << "Error: Cannot open file " << argv[1] << endl;
        return 1;
    }

    int value;
    while (file.read(reinterpret_cast<char*>(&value), sizeof(int))) {
        cout << value << " ";
    }
    cout << endl;

    file.close();
    return 0;
}
