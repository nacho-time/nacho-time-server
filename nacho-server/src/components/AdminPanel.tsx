import { createAsync, useSubmission, useAction } from "@solidjs/router";
import { For, Show, createSignal } from "solid-js";
import {
  getUsers,
  getSettings,
  removeUser,
  resetPassword,
  toggleAdmin,
  updateSettings,
} from "~/lib";

// Icon components
const UserIcon = (props: any) => (
  <svg {...props} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      stroke-linecap="round"
      stroke-linejoin="round"
      stroke-width="2"
      d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
    />
  </svg>
);

const TrashIcon = (props: any) => (
  <svg {...props} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      stroke-linecap="round"
      stroke-linejoin="round"
      stroke-width="2"
      d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
    />
  </svg>
);

const KeyIcon = (props: any) => (
  <svg {...props} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      stroke-linecap="round"
      stroke-linejoin="round"
      stroke-width="2"
      d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z"
    />
  </svg>
);

const ShieldIcon = (props: any) => (
  <svg {...props} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      stroke-linecap="round"
      stroke-linejoin="round"
      stroke-width="2"
      d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
    />
  </svg>
);

const SettingsIcon = (props: any) => (
  <svg {...props} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      stroke-linecap="round"
      stroke-linejoin="round"
      stroke-width="2"
      d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
    />
    <path
      stroke-linecap="round"
      stroke-linejoin="round"
      stroke-width="2"
      d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
    />
  </svg>
);

export default function AdminPanel() {
  const users = createAsync(() => getUsers());
  const settings = createAsync(() => getSettings());

  const [resetUserId, setResetUserId] = createSignal<string | null>(null);
  const [newPassword, setNewPassword] = createSignal("");

  const removeUserAction = useAction(removeUser);
  const resetPasswordAction = useAction(resetPassword);
  const toggleAdminAction = useAction(toggleAdmin);
  const updateSettingsAction = useAction(updateSettings);

  const removeSubmission = useSubmission(removeUser);
  const resetSubmission = useSubmission(resetPassword);
  const toggleSubmission = useSubmission(toggleAdmin);
  const settingsSubmission = useSubmission(updateSettings);

  const handleRemoveUser = async (userId: string, username: string) => {
    if (confirm(`Are you sure you want to delete user "${username}"?`)) {
      const result = await removeUserAction(new FormData());
      const formData = new FormData();
      formData.append("userId", userId);
      await removeUserAction(formData);
    }
  };

  const handleResetPassword = async (userId: string) => {
    if (newPassword().length < 6) {
      alert("Password must be at least 6 characters long");
      return;
    }

    const formData = new FormData();
    formData.append("userId", userId);
    formData.append("newPassword", newPassword());

    const result = await resetPasswordAction(formData);
    if (result && "success" in result && result.success) {
      alert("Password reset successfully!");
      setResetUserId(null);
      setNewPassword("");
    } else {
      alert(result?.error || "Failed to reset password");
    }
  };

  const handleToggleAdmin = async (userId: string, username: string) => {
    const formData = new FormData();
    formData.append("userId", userId);
    await toggleAdminAction(formData);
  };

  const handleToggleRegistration = async () => {
    const formData = new FormData();
    formData.append(
      "allowNewUserRegistration",
      String(!settings()?.allowNewUserRegistration)
    );
    await updateSettingsAction(formData);
  };

  return (
    <div class="bg-white rounded-2xl shadow-2xl overflow-hidden">
      {/* Header */}
      <div class="bg-gradient-to-r from-red-600 to-pink-600 px-6 md:px-8 py-6">
        <div class="flex items-center gap-3">
          <ShieldIcon class="w-8 h-8 text-white" />
          <div>
            <h2 class="text-2xl md:text-3xl font-bold text-white">
              Admin Panel
            </h2>
            <p class="text-red-100 mt-1">Manage users and system settings</p>
          </div>
        </div>
      </div>

      <div class="p-6 md:p-8 space-y-8">
        {/* System Settings */}
        <div class="border-b pb-6">
          <div class="flex items-center gap-2 mb-4">
            <SettingsIcon class="w-5 h-5 text-gray-700" />
            <h3 class="text-lg font-semibold text-gray-800">System Settings</h3>
          </div>
          <div class="bg-gray-50 rounded-lg p-4">
            <label class="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={settings()?.allowNewUserRegistration ?? true}
                onChange={handleToggleRegistration}
                disabled={settingsSubmission.pending}
                class="w-5 h-5 text-indigo-600 rounded focus:ring-2 focus:ring-indigo-500"
              />
              <div>
                <div class="font-medium text-gray-900">
                  Allow New User Registration
                </div>
                <div class="text-sm text-gray-500">
                  When disabled, only admins can create new accounts
                </div>
              </div>
            </label>
          </div>
        </div>

        {/* Users List */}
        <div>
          <div class="flex items-center gap-2 mb-4">
            <UserIcon class="w-5 h-5 text-gray-700" />
            <h3 class="text-lg font-semibold text-gray-800">User Management</h3>
          </div>

          <Show
            when={users() && users()!.length > 0}
            fallback={
              <div class="text-center py-8 text-gray-500">No users found</div>
            }
          >
            <div class="space-y-3">
              <For each={users()}>
                {(user) => (
                  <div class="bg-gray-50 rounded-lg p-4 hover:bg-gray-100 transition-colors">
                    <div class="flex items-center justify-between">
                      <div class="flex items-center gap-3">
                        <UserIcon class="w-6 h-6 text-gray-600" />
                        <div>
                          <div class="flex items-center gap-2">
                            <span class="font-semibold text-gray-900">
                              {user.username}
                            </span>
                            {user.isAdmin && (
                              <span class="px-2 py-0.5 bg-red-100 text-red-700 text-xs font-bold rounded">
                                ADMIN
                              </span>
                            )}
                          </div>
                          <div class="text-xs text-gray-500 mt-1">
                            {user._count.movieEntries} movies •{" "}
                            {user._count.episodeEntries} episodes •{" "}
                            {user._count.authTokens} tokens
                          </div>
                        </div>
                      </div>

                      <div class="flex items-center gap-2">
                        {/* Reset Password */}
                        <Show
                          when={resetUserId() === user.id}
                          fallback={
                            <button
                              onClick={() => setResetUserId(user.id)}
                              class="p-2 text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                              title="Reset Password"
                            >
                              <KeyIcon class="w-5 h-5" />
                            </button>
                          }
                        >
                          <div class="flex items-center gap-2">
                            <input
                              type="password"
                              value={newPassword()}
                              onInput={(e) => setNewPassword(e.target.value)}
                              placeholder="New password"
                              class="px-3 py-1 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                            />
                            <button
                              onClick={() => handleResetPassword(user.id)}
                              disabled={resetSubmission.pending}
                              class="px-3 py-1 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 disabled:opacity-50"
                            >
                              Save
                            </button>
                            <button
                              onClick={() => {
                                setResetUserId(null);
                                setNewPassword("");
                              }}
                              class="px-3 py-1 bg-gray-300 text-gray-700 text-sm rounded-lg hover:bg-gray-400"
                            >
                              Cancel
                            </button>
                          </div>
                        </Show>

                        {/* Toggle Admin */}
                        <button
                          onClick={() =>
                            handleToggleAdmin(user.id, user.username)
                          }
                          disabled={toggleSubmission.pending}
                          class={`p-2 rounded-lg transition-colors ${
                            user.isAdmin
                              ? "text-red-600 hover:bg-red-50"
                              : "text-green-600 hover:bg-green-50"
                          }`}
                          title={user.isAdmin ? "Remove Admin" : "Make Admin"}
                        >
                          <ShieldIcon class="w-5 h-5" />
                        </button>

                        {/* Delete User */}
                        <button
                          onClick={() =>
                            handleRemoveUser(user.id, user.username)
                          }
                          disabled={removeSubmission.pending}
                          class="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors disabled:opacity-50"
                          title="Delete User"
                        >
                          <TrashIcon class="w-5 h-5" />
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </For>
            </div>
          </Show>
        </div>
      </div>
    </div>
  );
}
