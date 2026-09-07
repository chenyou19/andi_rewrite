using System;
using System.Diagnostics;
using System.IO;

internal static class SynthStripLauncher
{
    public static int Main(string[] args)
    {
        string root = AppContext.BaseDirectory;
        string pythonFile = Path.Combine(root, "python_path.txt");
        string script = Path.Combine(root, "mri_synthstrip.py");

        if (!File.Exists(pythonFile) || !File.Exists(script))
        {
            Console.Error.WriteLine(
                "SynthStrip launcher is incomplete: expected python_path.txt and " +
                "mri_synthstrip.py beside the executable."
            );
            return 2;
        }

        string python = File.ReadAllText(pythonFile).Trim();
        if (python.Length == 0 || !File.Exists(python))
        {
            Console.Error.WriteLine("Configured Python executable does not exist: " + python);
            return 2;
        }

        var start = new ProcessStartInfo
        {
            FileName = python,
            UseShellExecute = false,
            WorkingDirectory = root,
        };
        start.Environment["FREESURFER_HOME"] = root.TrimEnd(Path.DirectorySeparatorChar);
        start.ArgumentList.Add(script);
        foreach (string arg in args)
        {
            start.ArgumentList.Add(arg);
        }

        using Process process = Process.Start(start);
        process.WaitForExit();
        return process.ExitCode;
    }
}
