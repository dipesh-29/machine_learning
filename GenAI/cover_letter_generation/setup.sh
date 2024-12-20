# setup.sh
#!/bin/bash

# Check current SQLite version
sqlite_version=$(sqlite3 --version | awk '{print $1}')
required_version="3.35.0"

# Function to compare version numbers
version_ge() {
    printf '%s\n%s' "$required_version" "$1" | sort -C -V
}

# Install if the version is less than the required version
if ! version_ge "$sqlite_version"; then
    echo "Updating SQLite to version $required_version..."
    sudo apt-get update
    sudo apt-get install -y sqlite3
else
    echo "SQLite version is sufficient: $sqlite_version"
fi
