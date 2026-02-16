from pathlib import Path

class PathFinder:
    @staticmethod
    def find(directory=None, suffix=None, prefix=None, file_and_directory=False):
        """
        Modernized file finder using pathlib.
        Returns a sorted list of matching Path objects.
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        # Normalize input parameters for suffix and prefix:
        if suffix is not None:
            if isinstance(suffix, str):
                ext_tuple = (suffix,)
            elif isinstance(suffix, list):
                ext_tuple = tuple(suffix)
            else:
                ext_tuple = suffix
        else:
            ext_tuple = None

        if prefix is not None:
            if isinstance(prefix, str):
                start_tuple = (prefix,)
            elif isinstance(prefix, list):
                start_tuple = tuple(prefix)
            else:
                start_tuple = prefix
        else:
            start_tuple = None

        matches = []
        for path in dir_path.rglob('*'):
            if not path.is_file():
                continue
            
            if 'saved_movies' in path.parts:
                    continue

            # Check file extension if provided.
            if ext_tuple and path.suffix not in ext_tuple:
                continue

            # Check filename prefix if provided.
            if start_tuple and not path.name.startswith(start_tuple):
                continue

            matches.append(path)

        if not matches:
            print(f"No files found matching criteria in {dir_path}. Returning None...")
            return

        # Sort by modification time.
        sorted_paths = sorted(matches, key=lambda p: p.stat().st_mtime)

        if file_and_directory:
            dirs = list({p.parent for p in sorted_paths})
            dirs.sort(key=lambda p: p.stat().st_mtime)
            return sorted_paths, dirs

        return sorted_paths
