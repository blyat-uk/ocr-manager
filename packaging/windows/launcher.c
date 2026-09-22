/* "OCR Manager.exe": starts the bundled interpreter on the bundled sources.
 *
 * A GUI-subsystem program, so no console window opens. It sets what every
 * launcher sets (see packaging/build.py): OCR_MANAGER_BUNDLE=<its folder>,
 * <its folder>\bin first on PATH (ffmpeg, ffprobe), then runs
 *
 *     python\pythonw.exe -s -E -B -X utf8 src\main.py <arguments>
 *
 * The arguments are passed through exactly as they were given: the rest of
 * the command line after the program name, so no re-quoting can go wrong.
 * It waits for the app and returns its exit code.
 *
 * Built by packaging/build.py with MSVC:
 *     rc launcher.rc && cl /DUNICODE launcher.c launcher.res /link /SUBSYSTEM:WINDOWS user32.lib
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdlib.h>
#include <wchar.h>

static int fail(const wchar_t *what, const wchar_t *detail)
{
    wchar_t text[2048];
    DWORD code = GetLastError();
    _snwprintf_s(text, _countof(text), _TRUNCATE,
                 L"%s\n\n%s\n\n(Windows error %lu)", what, detail ? detail : L"", code);
    MessageBoxW(NULL, text, L"OCR Manager", MB_ICONERROR | MB_OK);
    return 1;
}

/* The command line after argv[0], by CommandLineToArgvW's rule for the
 * program name: a quoted run up to the closing quote, else up to whitespace. */
static const wchar_t *arguments(const wchar_t *cmd)
{
    if (*cmd == L'"') {
        for (cmd++; *cmd && *cmd != L'"'; cmd++)
            ;
        if (*cmd)
            cmd++;
    } else {
        while (*cmd && *cmd != L' ' && *cmd != L'\t')
            cmd++;
    }
    while (*cmd == L' ' || *cmd == L'\t')
        cmd++;
    return cmd;
}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR cmdline, int show)
{
    wchar_t root[32768];
    DWORD n = GetModuleFileNameW(NULL, root, _countof(root));
    wchar_t *slash;
    (void)instance; (void)previous; (void)cmdline; (void)show;

    if (n == 0 || n >= _countof(root))
        return fail(L"Cannot find the application's folder.", NULL);
    slash = wcsrchr(root, L'\\');
    if (slash)
        *slash = L'\0';

    SetEnvironmentVariableW(L"OCR_MANAGER_BUNDLE", root);

    {   /* PATH = <root>\bin;<PATH> */
        DWORD old = GetEnvironmentVariableW(L"PATH", NULL, 0);
        size_t size = wcslen(root) + 6 + old + 1;
        wchar_t *path = (wchar_t *)malloc(size * sizeof(wchar_t));
        if (!path)
            return fail(L"Out of memory.", NULL);
        _snwprintf_s(path, size, _TRUNCATE, L"%s\\bin;", root);
        if (old)
            GetEnvironmentVariableW(L"PATH", path + wcslen(path), old);
        SetEnvironmentVariableW(L"PATH", path);
        free(path);
    }

    {
        const wchar_t *args = arguments(GetCommandLineW());
        size_t size = 3 * wcslen(root) + wcslen(args) + 128;
        wchar_t *python = (wchar_t *)malloc(size * sizeof(wchar_t));
        wchar_t *command = (wchar_t *)malloc(size * sizeof(wchar_t));
        STARTUPINFOW si;
        PROCESS_INFORMATION pi;
        DWORD code = 1;

        if (!python || !command)
            return fail(L"Out of memory.", NULL);
        _snwprintf_s(python, size, _TRUNCATE, L"%s\\python\\pythonw.exe", root);
        _snwprintf_s(command, size, _TRUNCATE,
                     L"\"%s\" -s -E -B -X utf8 \"%s\\src\\main.py\" %s", python, root, args);

        ZeroMemory(&si, sizeof(si));
        si.cb = sizeof(si);
        ZeroMemory(&pi, sizeof(pi));
        /* Let the app's window take the foreground this launcher was given. */
        AllowSetForegroundWindow(ASFW_ANY);
        if (!CreateProcessW(python, command, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi))
            return fail(L"Cannot start the bundled Python interpreter.", python);
        WaitForSingleObject(pi.hProcess, INFINITE);
        GetExitCodeProcess(pi.hProcess, &code);
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        free(python);
        free(command);
        return (int)code;
    }
}
