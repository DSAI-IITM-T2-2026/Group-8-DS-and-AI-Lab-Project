export interface AppConfig { apiBaseUrl: string }

export const appConfig: AppConfig = {
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL || "/api/v1",
};
