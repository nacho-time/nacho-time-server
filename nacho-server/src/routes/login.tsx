import {
  useSubmission,
  type RouteSectionProps,
  createAsync,
} from "@solidjs/router";
import { Show, createSignal } from "solid-js";
import { loginUser, registerUser, getSettings } from "~/lib";

export default function Login(props: RouteSectionProps) {
  const loggingIn = useSubmission(loginUser);
  const registering = useSubmission(registerUser);
  const settings = createAsync(() => getSettings());
  const [loginType, setLoginType] = createSignal<"login" | "register">("login");

  const isLoading = () => loggingIn.pending || registering.pending;
  const currentAction = () =>
    loginType() === "login" ? loginUser : registerUser;
  const currentResult = () =>
    loginType() === "login" ? loggingIn.result : registering.result;

  return (
    <main class="min-h-screen flex items-center justify-center bg-gradient-to-br from-indigo-100 via-purple-50 to-pink-100 px-4 py-12">
      <div class="w-full max-w-md">
        {/* Card Container */}
        <div class="bg-white rounded-2xl shadow-2xl overflow-hidden">
          {/* Header */}
          <div class="bg-gradient-to-r from-indigo-600 to-purple-600 px-8 py-6">
            <h1 class="text-3xl font-bold text-white text-center">
              🌮 Nacho Time
            </h1>
            <p class="text-indigo-100 text-center mt-2 text-sm">
              Welcome back! Please sign in to continue
            </p>
          </div>

          {/* Form Container */}
          <div class="px-8 py-8">
            <form action={currentAction()} method="post" class="space-y-6">
              {/* Login/Register Toggle */}
              <div class="flex items-center justify-center space-x-4 bg-gray-100 rounded-lg p-1">
                <label class="flex-1 cursor-pointer">
                  <input
                    type="radio"
                    name="loginType"
                    value="login"
                    checked={loginType() === "login"}
                    onChange={() => setLoginType("login")}
                    class="sr-only peer"
                  />
                  <div class="text-center py-2 px-4 rounded-md font-medium transition-all peer-checked:bg-white peer-checked:text-indigo-600 peer-checked:shadow-sm text-gray-600">
                    Login
                  </div>
                </label>
                <label class="flex-1 cursor-pointer">
                  <input
                    type="radio"
                    name="loginType"
                    value="register"
                    checked={loginType() === "register"}
                    onChange={() => setLoginType("register")}
                    class="sr-only peer"
                    disabled={!settings()?.allowNewUserRegistration}
                  />
                  <div
                    class={`text-center py-2 px-4 rounded-md font-medium transition-all ${
                      settings()?.allowNewUserRegistration
                        ? "peer-checked:bg-white peer-checked:text-indigo-600 peer-checked:shadow-sm text-gray-600"
                        : "text-gray-400 cursor-not-allowed"
                    }`}
                  >
                    Register
                  </div>
                </label>
              </div>

              {/* Registration Disabled Message */}
              <Show
                when={
                  loginType() === "register" &&
                  !settings()?.allowNewUserRegistration
                }
              >
                <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                  <div class="flex items-start gap-3">
                    <svg
                      class="w-5 h-5 text-yellow-600 mt-0.5"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        stroke-width="2"
                        d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                      />
                    </svg>
                    <div>
                      <h3 class="font-semibold text-yellow-800">
                        Registration Disabled
                      </h3>
                      <p class="text-sm text-yellow-700 mt-1">
                        New user registration is currently disabled. Please
                        contact an administrator.
                      </p>
                    </div>
                  </div>
                </div>
              </Show>

              {/* Username Input */}
              <div class="space-y-2">
                <label
                  for="username-input"
                  class="block text-sm font-medium text-gray-700"
                >
                  Username
                </label>
                <div class="relative">
                  <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <svg
                      class="h-5 w-5 text-gray-400"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        stroke-width="2"
                        d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                      />
                    </svg>
                  </div>
                  <input
                    id="username-input"
                    name="username"
                    type="text"
                    placeholder="Enter your username"
                    class="block w-full pl-10 pr-3 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all placeholder-gray-400"
                    required
                  />
                </div>
              </div>

              {/* Password Input */}
              <div class="space-y-2">
                <label
                  for="password-input"
                  class="block text-sm font-medium text-gray-700"
                >
                  Password
                </label>
                <div class="relative">
                  <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <svg
                      class="h-5 w-5 text-gray-400"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        stroke-width="2"
                        d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                      />
                    </svg>
                  </div>
                  <input
                    id="password-input"
                    name="password"
                    type="password"
                    placeholder="Enter your password"
                    class="block w-full pl-10 pr-3 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all placeholder-gray-400"
                    required
                  />
                </div>
              </div>

              {/* Error Message */}
              <Show when={currentResult()}>
                <div class="rounded-lg bg-red-50 border border-red-200 p-4">
                  <div class="flex items-center">
                    <svg
                      class="h-5 w-5 text-red-400 mr-2"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fill-rule="evenodd"
                        d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                        clip-rule="evenodd"
                      />
                    </svg>
                    <p
                      class="text-sm text-red-800"
                      role="alert"
                      id="error-message"
                    >
                      {(() => {
                        const result = currentResult();
                        if (typeof result === "string") {
                          return result;
                        } else if (result instanceof Error) {
                          return result.message;
                        }
                        return "An error occurred";
                      })()}
                    </p>
                  </div>
                </div>
              </Show>

              {/* Submit Button */}
              <button
                type="submit"
                class="w-full bg-gradient-to-r from-indigo-600 to-purple-600 text-white py-3 px-4 rounded-lg font-semibold hover:from-indigo-700 hover:to-purple-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-all transform hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed"
                disabled={isLoading()}
              >
                <Show
                  when={!isLoading()}
                  fallback={
                    <span class="flex items-center justify-center">
                      <svg
                        class="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                        fill="none"
                        viewBox="0 0 24 24"
                      >
                        <circle
                          class="opacity-25"
                          cx="12"
                          cy="12"
                          r="10"
                          stroke="currentColor"
                          stroke-width="4"
                        ></circle>
                        <path
                          class="opacity-75"
                          fill="currentColor"
                          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                        ></path>
                      </svg>
                      Processing...
                    </span>
                  }
                >
                  {loginType() === "login" ? "Sign In" : "Create Account"}
                </Show>
              </button>
            </form>
          </div>

          {/* Footer */}
          <div class="px-8 py-4 bg-gray-50 border-t border-gray-200">
            <p class="text-xs text-center text-gray-600">
              {loginType() === "login"
                ? "Don't have an account? "
                : "Already have an account? "}
              <button
                type="button"
                onClick={() => {
                  setLoginType(loginType() === "login" ? "register" : "login");
                }}
                class="text-indigo-600 hover:text-indigo-800 font-medium"
              >
                {loginType() === "login" ? "Register here" : "Sign in here"}
              </button>
            </p>
          </div>
        </div>

        {/* Additional Info */}
        <p class="text-center text-sm text-gray-600 mt-6">
          Secure login with Argon2id password hashing 🔒 Always check if https
          is used to protect your credentials.
        </p>
      </div>
    </main>
  );
}
