// LD_PRELOAD shim: when AdaptiveCpp spawns clspv, run $CLSPV_WRAPPER instead,
// so that extra clspv flags can be tried without rebuilding AdaptiveCpp.
// build: gcc -O2 -shared -fPIC clspv_shim.c -o clspv_shim.so -ldl
#define _GNU_SOURCE
#include <dlfcn.h>
#include <spawn.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *redirect(const char *path) {
  const char *w = getenv("CLSPV_WRAPPER");
  size_t n = strlen(path);
  if (w && n >= 6 && strcmp(path + n - 6, "/clspv") == 0) return w;
  return path;
}

int posix_spawn(pid_t *pid, const char *path, const posix_spawn_file_actions_t *fa,
                const posix_spawnattr_t *attr, char *const argv[], char *const envp[]) {
  static int (*real)(pid_t *, const char *, const posix_spawn_file_actions_t *,
                     const posix_spawnattr_t *, char *const[], char *const[]);
  if (!real) real = dlsym(RTLD_NEXT, "posix_spawn");
  return real(pid, redirect(path), fa, attr, argv, envp);
}

int execve(const char *path, char *const argv[], char *const envp[]) {
  static int (*real)(const char *, char *const[], char *const[]);
  if (!real) real = dlsym(RTLD_NEXT, "execve");
  return real(redirect(path), argv, envp);
}
