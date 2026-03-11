declare global {
  namespace NodeJS {
    interface Process {
      env: Record<string, string | undefined>;
      exit(code?: number): never;
    }
  }
}

export {};